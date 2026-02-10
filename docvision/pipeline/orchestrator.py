"""
Main pipeline orchestrator for document processing.

Coordinates all processing stages:
1. Input loading (PDF/image)
2. Preprocessing
3. Layout detection
4. Text detection
5. Table detection
6. OCR recognition
7. KIE extraction (Donut + LayoutLMv3)
8. Rank and fuse
9. Validation
10. Output generation
"""

import os
import time
import uuid
from pathlib import Path
from typing import Optional, List, Union, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import numpy as np
from loguru import logger

from docvision.config import Config, load_config
from docvision.types import (
    Document, Page, PageMetadata, Field, Table, TextLine,
    LayoutRegion, ProcessingResult, DocumentMetadata,
    ValidationResult, ContentType, FieldStatus
)


@dataclass
class ProcessingOptions:
    """Options for document processing."""
    # Processing mode: "local", "azure", or "hybrid"
    processing_mode: str = "local"

    # Processing stages (used in local/hybrid mode)
    preprocess: bool = True
    detect_layout: bool = True
    detect_text: bool = True
    detect_tables: bool = True
    run_ocr: bool = True
    run_donut: bool = True
    run_layoutlmv3: bool = True
    run_validators: bool = True

    # Azure-specific options (used in azure/hybrid mode)
    use_gpt_vision_kie: bool = True  # Use GPT-4o for field extraction
    document_type: str = "auto"      # auto, bol, invoice, receipt, delivery_ticket

    # Output options
    save_artifacts: bool = True
    save_json: bool = True
    output_dir: Optional[str] = None


class DocumentProcessor:
    """
    Main document processing pipeline.
    
    Orchestrates all stages of document understanding from
    raw input to structured JSON output.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize document processor.
        
        Args:
            config: Configuration object (uses defaults if None)
        """
        self.config = config or Config()
        self.device = self.config.runtime.get_device()
        
        # Initialize components lazily
        self._pdf_loader = None
        self._image_loader = None
        self._artifact_manager = None
        self._layout_detector = None
        self._text_detector = None
        self._table_detector = None
        self._trocr = None
        self._tesseract = None
        self._donut = None
        self._layoutlmv3 = None
        self._fuser = None

        # Azure cloud providers (lazy)
        self._azure_di_provider = None
        self._gpt_vision_extractor = None

        # Azure cost tracking and response caching
        self._cost_tracker = None
        self._response_cache = None
        
        logger.info(f"DocumentProcessor initialized with device: {self.device}")
    
    @property
    def pdf_loader(self):
        """Lazy load PDF loader."""
        if self._pdf_loader is None:
            from docvision.io.pdf import PDFLoader
            self._pdf_loader = PDFLoader(
                dpi=self.config.pdf.dpi,
                max_pages=self.config.pdf.max_pages
            )
        return self._pdf_loader
    
    @property
    def image_loader(self):
        """Lazy load image loader."""
        if self._image_loader is None:
            from docvision.io.image import ImageLoader
            self._image_loader = ImageLoader()
        return self._image_loader
    
    @property
    def artifact_manager(self):
        """Lazy load artifact manager."""
        if self._artifact_manager is None:
            from docvision.io.artifacts import ArtifactManager
            self._artifact_manager = ArtifactManager(
                output_dir=self.config.artifacts.dir,
                enable=self.config.artifacts.enable,
                save_layout=self.config.artifacts.save_layout_overlay,
                save_text_polygons=self.config.artifacts.save_text_polygons,
                save_table_structure=self.config.artifacts.save_table_structure,
                save_ocr_overlay=self.config.artifacts.save_ocr_overlay,
                save_preprocessed=self.config.artifacts.save_preprocessed
            )
        return self._artifact_manager
    
    @property
    def layout_detector(self):
        """Lazy load layout detector."""
        if self._layout_detector is None:
            from docvision.detect.layout_doclaynet import LayoutDetector
            self._layout_detector = LayoutDetector(
                model_path=self.config.models.layout,
                device=self.device
            )
        return self._layout_detector
    
    @property
    def text_detector(self):
        """Lazy load text detector."""
        if self._text_detector is None:
            from docvision.detect.text_craft import TextDetector
            self._text_detector = TextDetector(
                model_path=self.config.models.craft,
                device=self.device
            )
        return self._text_detector
    
    @property
    def table_detector(self):
        """Lazy load table detector."""
        if self._table_detector is None:
            from docvision.detect.table_tatr import TableDetector
            self._table_detector = TableDetector(
                model_name=self.config.models.tatr,
                device=self.device
            )
        return self._table_detector
    
    @property
    def trocr(self):
        """Lazy load TrOCR."""
        if self._trocr is None:
            from docvision.ocr.trocr import TrOCRRecognizer
            self._trocr = TrOCRRecognizer(
                printed_model=self.config.models.trocr_printed,
                handwritten_model=self.config.models.trocr_handwritten,
                device=self.device
            )
        return self._trocr
    
    @property
    def tesseract(self):
        """Lazy load Tesseract."""
        if self._tesseract is None:
            try:
                from docvision.ocr.tesseract import TesseractRecognizer
                self._tesseract = TesseractRecognizer()
            except Exception as e:
                logger.warning(f"Tesseract not available: {e}")
                self._tesseract = False  # Sentinel: don't retry
        return self._tesseract if self._tesseract is not False else None
    
    @property
    def donut(self):
        """Lazy load Donut."""
        if self._donut is None:
            from docvision.kie.donut_runner import DonutRunner
            self._donut = DonutRunner(
                model_name=self.config.models.donut,
                device=self.device
            )
        return self._donut
    
    @property
    def layoutlmv3(self):
        """Lazy load LayoutLMv3."""
        if self._layoutlmv3 is None:
            from docvision.kie.layoutlmv3_runner import LayoutLMv3Runner
            self._layoutlmv3 = LayoutLMv3Runner(
                model_name=self.config.models.layoutlmv3,
                device=self.device
            )
        return self._layoutlmv3
    
    @property
    def fuser(self):
        """Lazy load rank-and-fuse engine."""
        if self._fuser is None:
            from docvision.kie.fuse import RankAndFuse, FusionStrategy
            from docvision.types import SourceEngine
            self._fuser = RankAndFuse(
                strategy=FusionStrategy.WEIGHTED_VOTE,
                source_weights={
                    "donut": self.config.kie.donut_weight,
                    "layoutlmv3": self.config.kie.layoutlmv3_weight,
                    "trocr": self.config.kie.ocr_weight,
                    "tesseract": self.config.kie.ocr_weight * 0.9,
                    SourceEngine.GPT_VISION: 1.2,
                    SourceEngine.AZURE_DOC_INTELLIGENCE: 1.0,
                }
            )
        return self._fuser

    @property
    def cost_tracker(self):
        """Lazy load cost tracker (shared across all Azure providers)."""
        if self._cost_tracker is None:
            from docvision.azure.cost_tracker import CostTracker
            self._cost_tracker = CostTracker()
        return self._cost_tracker

    @property
    def response_cache(self):
        """Lazy load response cache (shared across all Azure providers)."""
        if self._response_cache is None:
            from docvision.azure.response_cache import ResponseCache
            self._response_cache = ResponseCache()
        return self._response_cache

    @property
    def azure_di_provider(self):
        """Lazy load Azure Document Intelligence provider."""
        if self._azure_di_provider is None:
            from docvision.azure.doc_intelligence import AzureDocIntelligenceProvider
            self._azure_di_provider = AzureDocIntelligenceProvider(
                self.config.azure,
                cost_tracker=self.cost_tracker,
                response_cache=self.response_cache,
            )
        return self._azure_di_provider

    @property
    def gpt_vision_extractor(self):
        """Lazy load GPT Vision KIE extractor."""
        if self._gpt_vision_extractor is None:
            from docvision.azure.gpt_vision_kie import GPTVisionExtractor
            self._gpt_vision_extractor = GPTVisionExtractor(
                self.config.azure,
                cost_tracker=self.cost_tracker,
                response_cache=self.response_cache,
            )
        return self._gpt_vision_extractor
    
    def process(
        self,
        input_path: str,
        options: Optional[ProcessingOptions] = None
    ) -> ProcessingResult:
        """
        Process a document and extract structured information.
        
        Args:
            input_path: Path to PDF or image file
            options: Processing options
            
        Returns:
            ProcessingResult with extracted document data
        """
        options = options or ProcessingOptions()
        start_time = time.time()
        doc_id = str(uuid.uuid4())[:8]
        
        logger.info(f"Processing document: {input_path} (ID: {doc_id})")
        
        try:
            # Determine file type and load
            path = Path(input_path)
            
            if not path.exists():
                return ProcessingResult(
                    success=False,
                    error=f"File not found: {input_path}"
                )
            
            file_type = self._get_file_type(path)
            
            if file_type == "pdf":
                page_images = self.pdf_loader.load(str(path))
            elif file_type == "image":
                page_images = [self.image_loader.load(str(path))]
            else:
                return ProcessingResult(
                    success=False,
                    error=f"Unsupported file type: {path.suffix}"
                )
            
            # Process each page
            pages = []
            all_tables = []
            all_fields_lists = []

            # Set mode on artifact manager so artifacts go under Local/ or Azure_Cloud/
            if options.save_artifacts and self.config.artifacts.enable:
                self.artifact_manager.current_mode = options.processing_mode

            # ── Azure batch optimisation: send entire PDF in one DI call ──
            if (
                options.processing_mode == "azure"
                and file_type == "pdf"
                and len(page_images) > 1
            ):
                return self._process_pdf_azure_batch(
                    path, page_images, doc_id, options, start_time
                )
            
            for page_num, page_image in enumerate(page_images, start=1):
                logger.info(f"Processing page {page_num}/{len(page_images)}")
                
                page_result = self._process_page(
                    page_image,
                    page_num,
                    doc_id,
                    options
                )
                
                pages.append(page_result["page"])
                all_tables.extend(page_result["tables"])
                all_fields_lists.append(page_result["fields"])
            
            # Fuse fields from all sources
            fused_fields = self.fuser.fuse_fields(all_fields_lists)
            
            # Run validators
            if options.run_validators:
                fused_fields = self._run_validators(fused_fields)
            
            # Create document
            processing_time = time.time() - start_time
            
            document = Document(
                id=doc_id,
                metadata=DocumentMetadata(
                    filename=path.name,
                    file_type=file_type,
                    file_size_bytes=path.stat().st_size,
                    processed_at=datetime.utcnow(),
                    processing_time_seconds=processing_time
                ),
                page_count=len(pages),
                pages=pages,
                tables=all_tables,
                fields=fused_fields,
                validation=self._summarize_validation(fused_fields)
            )
            
            # Save output
            if options.save_json:
                output_dir = options.output_dir or self.config.output.dir
                self._save_output(document, output_dir, options.processing_mode)
            
            # Generate artifact summary
            if options.save_artifacts and self.config.artifacts.enable:
                self.artifact_manager.generate_summary_html(document, doc_id)
            
            logger.info(f"Document processed successfully in {processing_time:.2f}s")
            
            return ProcessingResult(
                success=True,
                document=document
            )
        
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            import traceback
            return ProcessingResult(
                success=False,
                error=str(e),
                error_details={"traceback": traceback.format_exc()}
            )
    
    def _get_file_type(self, path: Path) -> str:
        """Determine file type from extension."""
        suffix = path.suffix.lower()
        
        if suffix == ".pdf":
            return "pdf"
        elif suffix in [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"]:
            return "image"
        else:
            return "unknown"
    
    def _process_page(
        self,
        image: np.ndarray,
        page_num: int,
        doc_id: str,
        options: ProcessingOptions
    ) -> Dict[str, Any]:
        """Process a single page."""
        # Route to Azure cloud pipeline when mode is "azure"
        if options.processing_mode == "azure":
            return self._process_page_azure(image, page_num, doc_id, options)

        h, w = image.shape[:2]
        
        # Preprocessing
        if options.preprocess:
            from docvision.preprocess.enhance import preprocess_for_ocr, detect_content_type, assess_readability
            
            processed = preprocess_for_ocr(
                image,
                denoise=self.config.preprocess.denoise,
                clahe=self.config.preprocess.clahe,
                sharpen=self.config.preprocess.sharpen,
                deskew=self.config.preprocess.deskew,
                dewarp=self.config.preprocess.dewarp
            )
            
            # Detect content type
            content_type, _ = detect_content_type(processed)
            readability, issues = assess_readability(processed)
            
            if options.save_artifacts:
                self.artifact_manager.save_preprocessed_image(
                    processed, doc_id, page_num, "preprocessed"
                )
        else:
            processed = image
            content_type = ContentType.UNKNOWN
            readability = "good"
            issues = []
        
        # Layout detection
        layout_regions = []
        if options.detect_layout:
            layout_regions = self.layout_detector.detect(processed)
            
            if options.save_artifacts:
                self.artifact_manager.save_layout_overlay(
                    processed, layout_regions, doc_id, page_num
                )
        
        # Text detection
        text_lines = []
        if options.detect_text:
            text_lines = self.text_detector.detect(processed)
            
            if options.save_artifacts:
                self.artifact_manager.save_text_polygons_overlay(
                    processed, text_lines, doc_id, page_num
                )
        
        # Table detection
        tables = []
        if options.detect_tables:
            # Detect tables in table regions from layout, or full page
            table_regions = self.layout_detector.get_tables(layout_regions)
            
            if table_regions:
                for region in table_regions:
                    detected = self.table_detector.detect(processed, region.bbox)
                    for t in detected:
                        t.page = page_num
                    tables.extend(detected)
            else:
                # Try full page table detection
                detected = self.table_detector.detect(processed)
                for t in detected:
                    t.page = page_num
                tables.extend(detected)
            
            if options.save_artifacts:
                self.artifact_manager.save_table_structure_overlay(
                    processed, tables, doc_id, page_num
                )
        
        # OCR recognition
        if options.run_ocr:
            # Recognize text lines
            text_lines = self.trocr.update_text_lines(
                processed, text_lines, content_type
            )
            
            # Recognize table cells
            for table in tables:
                self._recognize_table_cells(processed, table, content_type)
            
            # Run Tesseract as backup for low-confidence lines
            if self.tesseract:
                text_lines = self._run_tesseract_backup(processed, text_lines)
            
            if options.save_artifacts:
                self.artifact_manager.save_ocr_overlay(
                    processed, text_lines, doc_id, page_num
                )
        
        # Build raw text
        raw_text = "\n".join(line.text for line in text_lines if line.text)
        
        # KIE extraction
        fields = []
        
        if options.run_donut and self.config.kie.use_donut:
            donut_fields = self.donut.extract_to_fields(processed, page_num)
            fields.extend(donut_fields)
        
        if options.run_layoutlmv3 and self.config.kie.use_layoutlmv3:
            lm_fields = self.layoutlmv3.extract_from_text_lines(
                processed, text_lines, page_num
            )
            fields.extend(lm_fields)
        
        # Create page object
        page = Page(
            number=page_num,
            metadata=PageMetadata(
                width=w,
                height=h,
                dpi=self.config.pdf.dpi,
                content_type=content_type,
                readability=readability,
                readability_issues=issues
            ),
            layout_regions=layout_regions,
            text_lines=text_lines,
            tables=tables,
            raw_text=raw_text
        )
        
        return {
            "page": page,
            "tables": tables,
            "fields": fields
        }

    def _process_pdf_azure_batch(
        self,
        pdf_path: Path,
        page_images: List[np.ndarray],
        doc_id: str,
        options: ProcessingOptions,
        start_time: float,
    ) -> ProcessingResult:
        """
        Process a multi-page PDF in a single Azure DI call.

        Instead of sending each rasterised page image individually, this sends
        the raw PDF bytes to ``analyze_bytes()`` — one API call for all pages.
        GPT Vision KIE is still done per-page (GPT requires images).
        """
        logger.info(
            f"Azure batch mode: sending entire PDF ({len(page_images)} pages) "
            "in a single Document Intelligence call"
        )

        pdf_bytes = pdf_path.read_bytes()
        # Pass pixel dimensions of each rasterised page so Azure DI can
        # scale inch-based PDF coordinates into pixel space.
        pixel_dims = [(img.shape[1], img.shape[0]) for img in page_images]
        batch_result = self.azure_di_provider.analyze_bytes(
            pdf_bytes, pixel_dimensions=pixel_dims
        )

        # analyze_bytes returns {"pages": [...]} for multi-page, or a single dict
        if "pages" in batch_result:
            per_page_results = batch_result["pages"]
        else:
            per_page_results = [batch_result]

        pages = []
        all_tables = []
        all_fields_lists = []

        for page_num, (page_data, page_image) in enumerate(
            zip(per_page_results, page_images), start=1
        ):
            logger.info(f"Processing page {page_num}/{len(page_images)} (batch)")

            h, w = page_image.shape[:2]

            # Reconstruct typed objects from cached/batch data
            text_lines = page_data.get("text_lines", [])
            tables = page_data.get("tables", [])
            layout_regions = page_data.get("layout_regions", [])
            raw_text = page_data.get("raw_text", "")

            # Save artifacts
            if options.save_artifacts:
                # Save the original page image as "preprocessed" so Azure
                # produces the same artifact set as the local pipeline.
                self.artifact_manager.save_preprocessed_image(
                    page_image, doc_id, page_num, "preprocessed"
                )
                self.artifact_manager.save_layout_overlay(
                    page_image, layout_regions, doc_id, page_num
                )
                self.artifact_manager.save_text_polygons_overlay(
                    page_image, text_lines, doc_id, page_num
                )
                self.artifact_manager.save_table_structure_overlay(
                    page_image, tables, doc_id, page_num
                )
                self.artifact_manager.save_ocr_overlay(
                    page_image, text_lines, doc_id, page_num
                )

            # GPT Vision KIE (still per-page — GPT needs images)
            fields = []
            if options.use_gpt_vision_kie and self.config.azure.is_openai_ready:
                fields = self.gpt_vision_extractor.extract(
                    page_image,
                    page_num=page_num,
                    ocr_text=raw_text,
                    document_type=options.document_type,
                )

            page = Page(
                number=page_num,
                metadata=PageMetadata(
                    width=w,
                    height=h,
                    dpi=self.config.pdf.dpi,
                    content_type=ContentType.UNKNOWN,
                    readability="good",
                    readability_issues=[],
                ),
                layout_regions=layout_regions,
                text_lines=text_lines,
                tables=tables,
                raw_text=raw_text,
            )

            pages.append(page)
            all_tables.extend(tables)
            all_fields_lists.append(fields)

        # Fuse fields from all sources
        fused_fields = self.fuser.fuse_fields(all_fields_lists)

        # Run validators
        if options.run_validators:
            fused_fields = self._run_validators(fused_fields)

        processing_time = time.time() - start_time

        document = Document(
            id=doc_id,
            metadata=DocumentMetadata(
                filename=pdf_path.name,
                file_type="pdf",
                file_size_bytes=pdf_path.stat().st_size,
                processed_at=datetime.utcnow(),
                processing_time_seconds=processing_time,
            ),
            page_count=len(pages),
            pages=pages,
            tables=all_tables,
            fields=fused_fields,
            validation=self._summarize_validation(fused_fields),
        )

        # Save output
        if options.save_json:
            output_dir = options.output_dir or self.config.output.dir
            self._save_output(document, output_dir, options.processing_mode)

        # Generate artifact summary
        if options.save_artifacts and self.config.artifacts.enable:
            self.artifact_manager.generate_summary_html(document, doc_id)

        logger.info(
            f"Batch document processed successfully in {processing_time:.2f}s "
            f"({len(pages)} pages)"
        )

        return ProcessingResult(success=True, document=document)

    def _process_page_azure(
        self,
        image: np.ndarray,
        page_num: int,
        doc_id: str,
        options: ProcessingOptions,
    ) -> Dict[str, Any]:
        """Process a single page using Azure cloud APIs."""
        h, w = image.shape[:2]
        logger.info(f"Processing page {page_num} via Azure cloud pipeline")

        # ── Azure Document Intelligence (OCR + layout + tables) ─────
        azure_result = self.azure_di_provider.analyze(image, page_num=page_num)

        text_lines = azure_result["text_lines"]
        tables = azure_result["tables"]
        layout_regions = azure_result["layout_regions"]
        raw_text = azure_result["raw_text"]

        # ── Save artifacts (same overlays as local pipeline) ────────
        if options.save_artifacts:
            # Save the original page image as "preprocessed" so Azure
            # produces the same artifact set as the local pipeline.
            self.artifact_manager.save_preprocessed_image(
                image, doc_id, page_num, "preprocessed"
            )
            self.artifact_manager.save_layout_overlay(
                image, layout_regions, doc_id, page_num
            )
            self.artifact_manager.save_text_polygons_overlay(
                image, text_lines, doc_id, page_num
            )
            self.artifact_manager.save_table_structure_overlay(
                image, tables, doc_id, page_num
            )
            self.artifact_manager.save_ocr_overlay(
                image, text_lines, doc_id, page_num
            )

        # ── GPT Vision KIE (field extraction) ───────────────────────
        fields = []
        if options.use_gpt_vision_kie and self.config.azure.is_openai_ready:
            fields = self.gpt_vision_extractor.extract(
                image,
                page_num=page_num,
                ocr_text=raw_text,
                document_type=options.document_type,
            )

        # ── Build page object ───────────────────────────────────────
        page = Page(
            number=page_num,
            metadata=PageMetadata(
                width=w,
                height=h,
                dpi=self.config.pdf.dpi,
                content_type=ContentType.UNKNOWN,
                readability="good",
                readability_issues=[],
            ),
            layout_regions=layout_regions,
            text_lines=text_lines,
            tables=tables,
            raw_text=raw_text,
        )

        return {
            "page": page,
            "tables": tables,
            "fields": fields,
        }

    def _recognize_table_cells(
        self,
        image: np.ndarray,
        table: Table,
        content_type: ContentType
    ) -> None:
        """Recognize text in table cells."""
        from docvision.ocr.crops import crop_text_region
        
        for cell in table.cells:
            if not cell.bbox:
                continue
            
            try:
                crop = crop_text_region(image, cell.bbox, padding=2)
                
                if crop.size > 0:
                    text, conf = self.trocr.recognize(crop, content_type)
                    cell.text = text
                    cell.confidence = conf
            except Exception as e:
                logger.debug(f"Failed to recognize cell: {e}")
    
    def _run_tesseract_backup(
        self,
        image: np.ndarray,
        text_lines: List[TextLine]
    ) -> List[TextLine]:
        """Run Tesseract on low-confidence lines for ensemble comparison."""
        from docvision.ocr.crops import crop_text_region
        from docvision.types import SourceEngine
        
        threshold = self.config.thresholds.reroute_to_tesseract_below
        rerouted_count = 0
        
        for line in text_lines:
            if line.confidence < threshold:
                try:
                    crop = crop_text_region(image, line.bbox, padding=4)
                    if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
                        continue
                    
                    tess_text, tess_conf = self.tesseract.recognize_line(crop)
                    
                    if tess_text.strip() and tess_conf > line.confidence:
                        logger.debug(
                            f"Tesseract improved: '{line.text}' ({line.confidence:.2f}) "
                            f"-> '{tess_text}' ({tess_conf:.2f})"
                        )
                        line.text = tess_text
                        line.confidence = tess_conf
                        line.source = SourceEngine.TESSERACT
                        rerouted_count += 1
                except Exception as e:
                    logger.debug(f"Tesseract backup failed for line: {e}")
        
        if rerouted_count > 0:
            logger.info(f"Tesseract improved {rerouted_count} low-confidence lines")
        
        return text_lines
    
    def _run_validators(self, fields: List[Field]) -> List[Field]:
        """Run validators on extracted fields."""
        from docvision.kie.validators import run_all_validators, validate_document_consistency
        
        for field in fields:
            results = run_all_validators(field)
            field.validators = results
            
            # Update status based on validation
            if results:
                all_passed = all(r.passed for r in results)
                if all_passed and field.confidence >= 0.5:
                    field.status = FieldStatus.VALIDATED
                elif not all_passed:
                    field.status = FieldStatus.VALIDATION_FAILED
        
        # Document-level consistency checks
        consistency_results = validate_document_consistency(fields)
        
        # Log any consistency issues
        for result in consistency_results:
            if not result.passed:
                logger.warning(f"Consistency check failed: {result.message}")
        
        return fields
    
    def _summarize_validation(self, fields: List[Field]) -> ValidationResult:
        """Summarize validation results across all fields."""
        total_checks = 0
        passed_checks = 0
        issues = []
        all_results = []
        
        for field in fields:
            for result in field.validators:
                total_checks += 1
                if result.passed:
                    passed_checks += 1
                else:
                    issues.append(f"{field.name}: {result.message}")
                all_results.append(result)
        
        return ValidationResult(
            passed=len(issues) == 0,
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=total_checks - passed_checks,
            issues=issues,
            details=all_results
        )
    
    def _save_output(self, document: Document, output_dir: str, processing_mode: str = "local") -> str:
        """Save document to JSON file."""
        output_path = Path(output_dir)
        # Route into Local/ or Azure_Cloud/ subfolder
        subfolder = "Azure_Cloud" if processing_mode == "azure" else "Local"
        output_path = output_path / subfolder
        output_path.mkdir(parents=True, exist_ok=True)
        
        filename = f"{document.metadata.filename}_{document.id}.json"
        filepath = output_path / filename
        
        # Convert to dict and serialize
        data = document.model_dump(mode="json")
        
        with open(filepath, "w", encoding="utf-8") as f:
            if self.config.output.pretty_json:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            else:
                json.dump(data, f, ensure_ascii=False, default=str)
        
        logger.info(f"Output saved to: {filepath}")
        return str(filepath)
    
    def process_batch(
        self,
        input_paths: List[str],
        options: Optional[ProcessingOptions] = None,
        parallel: bool = False
    ) -> List[ProcessingResult]:
        """
        Process multiple documents.
        
        Args:
            input_paths: List of file paths
            options: Processing options
            parallel: Whether to process in parallel
            
        Returns:
            List of ProcessingResults
        """
        results = []
        
        if parallel:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            workers = self.config.runtime.get_workers()
            
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(self.process, path, options): path
                    for path in input_paths
                }
                
                for future in as_completed(futures):
                    path = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Failed to process {path}: {e}")
                        results.append(ProcessingResult(
                            success=False,
                            error=str(e)
                        ))
        else:
            for path in input_paths:
                result = self.process(path, options)
                results.append(result)
        
        return results

    # ── Cost / cache stats ───────────────────────────────────────────────

    def get_cost_stats(self) -> Dict[str, Any]:
        """Return combined cost tracker + cache statistics."""
        stats: Dict[str, Any] = {}

        if self._cost_tracker is not None:
            stats["costs"] = self.cost_tracker.to_dict()
        else:
            stats["costs"] = {"total_calls": 0, "estimated_cost_usd": 0}

        if self._response_cache is not None:
            stats["cache"] = self.response_cache.stats()
        else:
            stats["cache"] = {"enabled": False}

        return stats

    def print_cost_summary(self) -> None:
        """Print a human-readable cost summary to the log."""
        if self._cost_tracker is not None:
            logger.info("\n" + self.cost_tracker.summary())
