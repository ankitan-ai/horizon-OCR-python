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
import threading
from pathlib import Path
from typing import Optional, List, Union, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import numpy as np
from loguru import logger

from docvision.config import Config, load_config
from docvision.io.markdown import save_markdown
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

    # Progress callback: called with (stage: str, percent: int)
    progress_callback: Optional[Callable[[str, int], None]] = None


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
        self._init_lock = threading.Lock()  # guards lazy component init
        
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
        self._style_extractor = None
        self._targeted_reocr = None

        # Azure cloud providers (lazy)
        self._azure_di_provider = None
        self._gpt_vision_extractor = None

        # Smart document classifier (lazy)
        self._document_classifier = None

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
            with self._init_lock:
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
            with self._init_lock:
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
            with self._init_lock:
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
    def style_extractor(self):
        """Lazy load style extractor for font/style information."""
        if self._style_extractor is None:
            from docvision.extract.pdf_style_extractor import StyleExtractor
            self._style_extractor = StyleExtractor()
        return self._style_extractor

    @property
    def targeted_reocr(self):
        """Lazy load targeted re-OCR processor for low-confidence regions."""
        if self._targeted_reocr is None:
            from docvision.ocr.targeted_reocr import TargetedReOCR, ReOCRConfig, ReOCRStrategy
            
            # Map string strategy from config to enum
            strategy_map = {
                "ensemble": ReOCRStrategy.ENSEMBLE,
                "trocr_only": ReOCRStrategy.TROCR_ONLY,
                "tesseract": ReOCRStrategy.TESSERACT_ONLY,
                "sequential": ReOCRStrategy.SEQUENTIAL,
            }
            strategy = strategy_map.get(
                self.config.reocr.strategy,
                ReOCRStrategy.ENSEMBLE
            )
            
            reocr_config = ReOCRConfig(
                confidence_threshold=self.config.reocr.confidence_threshold,
                improvement_threshold=self.config.reocr.improvement_threshold,
                strategy=strategy,
                max_reocr_lines=self.config.reocr.max_lines_per_page,
                scale_factor=self.config.reocr.scale_factor,
                enhanced_denoise_strength=self.config.reocr.enhanced_denoise,
                enhanced_clahe_clip=self.config.reocr.enhanced_clahe,
                enhanced_sharpen_strength=self.config.reocr.enhanced_sharpen,
                apply_binarization=self.config.reocr.apply_binarization,
                apply_morphology=self.config.reocr.apply_morphology,
                azure_retry_enabled=self.config.reocr.azure_retry_enabled,
                azure_retry_threshold=self.config.reocr.azure_retry_threshold,
            )
            
            self._targeted_reocr = TargetedReOCR(
                config=reocr_config,
                trocr_recognizer=self._trocr,  # Share existing recognizers
                tesseract_recognizer=self._tesseract,
                device=self.device
            )
        return self._targeted_reocr

    @property
    def cost_tracker(self):
        """Lazy load cost tracker (shared across all Azure providers)."""
        if self._cost_tracker is None:
            from docvision.azure.cost_tracker import CostTracker
            self._cost_tracker = CostTracker()
        return self._cost_tracker

    @property
    def has_cost_tracker(self) -> bool:
        """Check if cost tracker has been initialised (without triggering lazy load)."""
        return self._cost_tracker is not None

    @property
    def response_cache(self):
        """Lazy load response cache (shared across all Azure providers)."""
        if self._response_cache is None:
            from docvision.azure.response_cache import ResponseCache
            self._response_cache = ResponseCache()
        return self._response_cache

    @property
    def has_response_cache(self) -> bool:
        """Check if response cache has been initialised (without triggering lazy load)."""
        return self._response_cache is not None

    @property
    def azure_di_provider(self):
        """Lazy load Azure Document Intelligence provider."""
        if self._azure_di_provider is None:
            with self._init_lock:
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
            with self._init_lock:
                if self._gpt_vision_extractor is None:
                    from docvision.azure.gpt_vision_kie import GPTVisionExtractor
                    self._gpt_vision_extractor = GPTVisionExtractor(
                        self.config.azure,
                        cost_tracker=self.cost_tracker,
                        response_cache=self.response_cache,
                    )
        return self._gpt_vision_extractor

    @property
    def document_classifier(self):
        """Lazy load smart document classifier."""
        if self._document_classifier is None:
            from docvision.azure.classifier import DocumentClassifier
            self._document_classifier = DocumentClassifier(
                self.config.azure,
                cost_tracker=self.cost_tracker,
            )
        return self._document_classifier

    # ── Smart routing ────────────────────────────────────────────────────

    def _classify_and_route(
        self,
        first_page: "np.ndarray",
        options: ProcessingOptions,
    ) -> tuple:
        """
        Run GPT-nano classification on the first page and determine
        the best GPT deployment and DI model for extraction.

        Returns:
            (document_type, gpt_deployment, di_model)
        """
        from docvision.azure.classifier import ClassificationResult

        routing_cfg = self.config.smart_routing

        # Skip classification if disabled or if user already specified a type
        if not routing_cfg.enable:
            return options.document_type, None, None

        if routing_cfg.classify_on_auto_only and options.document_type != "auto":
            return options.document_type, None, None

        if not self.config.azure.is_openai_ready:
            return options.document_type, None, None

        logger.info("Running smart document classification (GPT-nano) …")
        result: ClassificationResult = self.document_classifier.classify(first_page)

        doc_type = result.document_type
        gpt_deploy = result.recommended_gpt_deployment
        di_model = result.recommended_di_model

        # Fall back to defaults when classifier returns nothing useful
        if doc_type == "auto":
            doc_type = options.document_type
        if not gpt_deploy:
            gpt_deploy = routing_cfg.default_gpt_deployment

        return doc_type, gpt_deploy, di_model

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
        _cb = options.progress_callback or (lambda stage, pct: None)
        _di_model_overridden = False
        _original_di_model = self.config.azure.doc_intelligence_model

        try:
            # Determine file type and load
            path = Path(input_path)
            # Use input filename stem as the folder / label for artifacts
            file_label = path.stem
            
            if not path.exists():
                return ProcessingResult(
                    success=False,
                    error=f"File not found: {input_path}"
                )
            
            _cb("Loading document", 5)
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

            # ── Smart classification (Azure mode only) ───────────────
            _cb("Classifying document", 10)
            gpt_deployment_override = None
            if options.processing_mode == "azure" and page_images:
                doc_type, gpt_deploy, di_model = self._classify_and_route(
                    page_images[0], options
                )
                if doc_type and doc_type != "auto":
                    options.document_type = doc_type
                if gpt_deploy:
                    gpt_deployment_override = gpt_deploy
                if di_model:
                    # Override DI model for this request only (thread-safe save/restore)
                    self.config.azure.doc_intelligence_model = di_model
                    _di_model_overridden = True

            # ── Azure batch optimisation: send entire PDF in one DI call ──
            if (
                options.processing_mode == "azure"
                and file_type == "pdf"
                and len(page_images) > 1
            ):
                return self._process_pdf_azure_batch(
                    path, page_images, doc_id, options, start_time,
                    file_label=file_label,
                    gpt_deployment_override=gpt_deployment_override,
                )
            
            for page_num, page_image in enumerate(page_images, start=1):
                base_pct = 15 + int(70 * (page_num - 1) / max(len(page_images), 1))
                _cb(f"Processing page {page_num}/{len(page_images)}", base_pct)
                logger.info(f"Processing page {page_num}/{len(page_images)}")
                
                page_result = self._process_page(
                    page_image,
                    page_num,
                    file_label,
                    options,
                    gpt_deployment_override=gpt_deployment_override,
                    pdf_path=path if file_type == "pdf" else None,
                )
                
                pages.append(page_result["page"])
                all_tables.extend(page_result["tables"])
                all_fields_lists.append(page_result["fields"])
            
            # Fuse fields from all sources
            _cb("Fusing fields", 88)
            fused_fields = self.fuser.fuse_fields(all_fields_lists)
            
            # Run validators
            if options.run_validators:
                _cb("Validating", 92)
                fused_fields = self._run_validators(fused_fields)
            
            # Create document
            processing_time = time.time() - start_time
            
            document = Document(
                id=doc_id,
                metadata=DocumentMetadata(
                    filename=path.name,
                    file_type=file_type,
                    file_size_bytes=path.stat().st_size,
                    processed_at=datetime.now(ZoneInfo("America/New_York")),
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
                self.artifact_manager.generate_summary_html(document, file_label)
            
            _cb("Complete", 100)
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
        finally:
            # Restore DI model if it was overridden by smart routing
            if _di_model_overridden:
                self.config.azure.doc_intelligence_model = _original_di_model
    
    def _get_file_type(self, path: Path) -> str:
        """Determine file type from extension."""
        suffix = path.suffix.lower()
        
        if suffix == ".pdf":
            return "pdf"
        elif suffix in [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"]:
            return "image"
        else:
            return "unknown"
    
    def _bboxes_overlap(self, bbox1: List[float], bbox2: List[float], threshold: float = 0.5) -> bool:
        """Check if two bounding boxes overlap significantly."""
        if not bbox1 or not bbox2 or len(bbox1) < 4 or len(bbox2) < 4:
            return False
        
        x1_min, y1_min, x1_max, y1_max = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
        x2_min, y2_min, x2_max, y2_max = bbox2[0], bbox2[1], bbox2[2], bbox2[3]
        
        # Compute intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return False
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        bbox1_area = max((x1_max - x1_min) * (y1_max - y1_min), 1)
        
        return (inter_area / bbox1_area) >= threshold

    def _estimate_text_line_styles(
        self, text_lines: List[Any], page_height: float, page_num: int
    ) -> None:
        """Estimate styles for text lines when PDF-native extraction isn't available."""
        from docvision.extract.pdf_style_extractor import estimate_style_from_bbox
        
        for line in text_lines:
            if not hasattr(line, 'bbox') or not line.bbox:
                continue
            
            bbox_height = line.bbox.y2 - line.bbox.y1
            y_position = line.bbox.y1
            
            # Get layout role if available
            role = getattr(line, 'role', None)
            
            estimated = estimate_style_from_bbox(
                text=line.text or "",
                bbox_height=bbox_height,
                y_position=y_position,
                page_height=page_height,
                role=role
            )
            line.style = estimated

    def _process_page(
        self,
        image: np.ndarray,
        page_num: int,
        doc_id: str,
        options: ProcessingOptions,
        gpt_deployment_override: Optional[str] = None,
        pdf_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Process a single page."""
        # Route to Azure cloud pipeline when mode is "azure"
        if options.processing_mode == "azure":
            return self._process_page_azure(
                image, page_num, doc_id, options,
                gpt_deployment_override=gpt_deployment_override,
            )

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
            
            # Targeted re-OCR for remaining low-confidence regions
            if self.config.reocr.enable and text_lines:
                text_lines = self.targeted_reocr.process_local(
                    processed, text_lines, content_type, in_place=True
                )
            
            if options.save_artifacts:
                self.artifact_manager.save_ocr_overlay(
                    processed, text_lines, doc_id, page_num
                )
        
        # Style extraction (PDF-native for digital PDFs, estimation for scanned/images)
        if text_lines and pdf_path:
            try:
                # extract_from_pdf returns (dict[page_num -> spans], source)
                styles_dict, source = self.style_extractor.extract_from_pdf(
                    pdf_path, page_numbers=[page_num]
                )
                styled_spans = styles_dict.get(page_num, [])
                
                if styled_spans:
                    # Update text_lines with style info from PDF
                    for line in text_lines:
                        if hasattr(line, 'bbox') and line.bbox:
                            line_bbox = [line.bbox.x1, line.bbox.y1, line.bbox.x2, line.bbox.y2]
                            matching_spans = [
                                s for s in styled_spans 
                                if self._bboxes_overlap(line_bbox, [s.x, s.y, s.x + s.width, s.y + s.height])
                            ]
                            if matching_spans:
                                # Use the most confident/common style for the line
                                best_span = max(matching_spans, key=lambda s: s.style.confidence)
                                line.style = best_span.style
                else:
                    # PDF-native failed or scanned - fall through to estimation
                    self._estimate_text_line_styles(text_lines, h, page_num)
            except Exception as e:
                logger.warning(f"Style extraction failed for page {page_num}: {e}")
                self._estimate_text_line_styles(text_lines, h, page_num)
        elif text_lines:
            # No PDF - estimate styles from bbox/text characteristics
            self._estimate_text_line_styles(text_lines, h, page_num)
        
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
        file_label: Optional[str] = None,
        gpt_deployment_override: Optional[str] = None,
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
        _cb = options.progress_callback or (lambda stage, pct: None)
        _cb("Sending to Azure Document Intelligence", 15)

        pdf_bytes = pdf_path.read_bytes()
        artifact_label = file_label or pdf_path.stem
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
            base_pct = 30 + int(50 * (page_num - 1) / max(len(page_images), 1))
            _cb(f"Processing page {page_num}/{len(page_images)}", base_pct)
            logger.info(f"Processing page {page_num}/{len(page_images)} (batch)")

            h, w = page_image.shape[:2]

            # ── Image quality metadata (no image mutation) ──────────
            try:
                from docvision.preprocess.enhance import detect_content_type, assess_readability
                page_content_type, _ = detect_content_type(page_image)
                page_readability, page_readability_issues = assess_readability(page_image)
            except Exception:
                page_content_type = ContentType.UNKNOWN
                page_readability = "good"
                page_readability_issues = []

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
                    page_image, artifact_label, page_num, "preprocessed"
                )
                self.artifact_manager.save_layout_overlay(
                    page_image, layout_regions, artifact_label, page_num
                )
                self.artifact_manager.save_text_polygons_overlay(
                    page_image, text_lines, artifact_label, page_num
                )
                self.artifact_manager.save_table_structure_overlay(
                    page_image, tables, artifact_label, page_num
                )
                self.artifact_manager.save_ocr_overlay(
                    page_image, text_lines, artifact_label, page_num
                )

            # GPT Vision KIE (still per-page — GPT needs images)
            fields = []
            if options.use_gpt_vision_kie and self.config.azure.is_openai_ready:
                fields = self.gpt_vision_extractor.extract(
                    page_image,
                    page_num=page_num,
                    ocr_text=raw_text,
                    document_type=options.document_type,
                    deployment_override=gpt_deployment_override,
                )

                # Anchor GPT Vision fields to Azure DI spatial coordinates
                self._anchor_fields_to_text(fields, text_lines, tables)

            # ── Style estimation for Azure batch pipeline ──────────────
            if text_lines:
                from docvision.extract.pdf_style_extractor import estimate_style_from_bbox
                for line in text_lines:
                    if not hasattr(line, 'bbox') or not line.bbox:
                        continue
                    bbox_height = line.bbox.y2 - line.bbox.y1
                    estimated = estimate_style_from_bbox(
                        text=line.text or "",
                        bbox_height=bbox_height,
                        y_position=line.bbox.y1,
                        page_height=h,
                        role=None,
                    )
                    line.style = estimated

            page = Page(
                number=page_num,
                metadata=PageMetadata(
                    width=w,
                    height=h,
                    dpi=self.config.pdf.dpi,
                    content_type=page_content_type,
                    readability=page_readability,
                    readability_issues=page_readability_issues,
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
        _cb("Fusing fields", 85)
        fused_fields = self.fuser.fuse_fields(all_fields_lists)

        # Run validators
        if options.run_validators:
            _cb("Validating", 90)
            fused_fields = self._run_validators(fused_fields)

        processing_time = time.time() - start_time

        document = Document(
            id=doc_id,
            metadata=DocumentMetadata(
                filename=pdf_path.name,
                file_type="pdf",
                file_size_bytes=pdf_path.stat().st_size,
                processed_at=datetime.now(ZoneInfo("America/New_York")),
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
            self.artifact_manager.generate_summary_html(document, artifact_label)

        _cb("Complete", 100)
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
        gpt_deployment_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process a single page using Azure cloud APIs."""
        h, w = image.shape[:2]
        logger.info(f"Processing page {page_num} via Azure cloud pipeline")

        # ── Image quality metadata (no image mutation) ──────────────
        try:
            from docvision.preprocess.enhance import detect_content_type, assess_readability
            content_type, _ = detect_content_type(image)
            readability, readability_issues = assess_readability(image)
        except Exception:
            content_type = ContentType.UNKNOWN
            readability = "good"
            readability_issues = []

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
                deployment_override=gpt_deployment_override,
            )

            # Anchor GPT Vision fields to Azure DI spatial coordinates
            self._anchor_fields_to_text(fields, text_lines, tables)

        # ── Style estimation for Azure pipeline ────────────────────
        # Azure DI doesn't provide font/style info directly, so we estimate
        # from bbox dimensions. LayoutRegions provide semantic roles for better estimation.
        if text_lines:
            # Build a role map from layout regions
            role_map = {}
            for region in layout_regions:
                if hasattr(region, 'text_lines'):
                    for tl in region.text_lines:
                        if hasattr(tl, 'id'):
                            role_map[tl.id] = region.type.value if hasattr(region.type, 'value') else str(region.type)
            
            # Estimate styles using bbox and role info
            for line in text_lines:
                if not hasattr(line, 'bbox') or not line.bbox:
                    continue
                
                role = role_map.get(line.id) if hasattr(line, 'id') else None
                bbox_height = line.bbox.y2 - line.bbox.y1
                
                from docvision.extract.pdf_style_extractor import estimate_style_from_bbox
                estimated = estimate_style_from_bbox(
                    text=line.text or "",
                    bbox_height=bbox_height,
                    y_position=line.bbox.y1,
                    page_height=h,
                    role=role
                )
                line.style = estimated

        # ── Build page object ───────────────────────────────────────
        page = Page(
            number=page_num,
            metadata=PageMetadata(
                width=w,
                height=h,
                dpi=self.config.pdf.dpi,
                content_type=content_type,
                readability=readability,
                readability_issues=readability_issues,
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
    
    # ── Spatial anchoring ───────────────────────────────────────────────

    @staticmethod
    def _anchor_fields_to_text(
        fields: List[Field],
        text_lines: List["TextLine"],
        tables: List["Table"],
    ) -> List[Field]:
        """
        Attach bounding boxes to fields that lack spatial coordinates.

        For every :class:`Field` (and its :class:`Candidate` entries) whose
        ``bbox`` is ``None``, search the Azure DI ``text_lines`` and ``tables``
        for a word/line/cell whose text matches the field value, and copy
        the matching bounding box.

        Match strategy (in priority order):
        1. **Exact word match** — iterate all ``Word`` objects inside every
           ``TextLine`` and pick the word whose ``text`` equals the normalised
           field value.  This gives the tightest possible box.
        2. **Exact line match** — if the field value equals a full
           ``TextLine.text`` (after stripping), use the line's bbox.
        3. **Substring / multi-word span** — if the field value appears as a
           contiguous substring inside a ``TextLine``, compute a merged bbox
           from the matching words.
        4. **Table cell match** — search ``Cell.text`` across all tables.
        5. If nothing matches, bbox stays ``None`` (no false anchoring).

        Returns:
            The same ``fields`` list (mutated in-place for convenience).
        """
        if not text_lines and not tables:
            return fields

        # Build a lookup: normalised word text → list of Word objects
        from docvision.types import BoundingBox

        word_index: Dict[str, List[Any]] = {}
        for tl in text_lines:
            for w in tl.words:
                key = w.text.strip().lower()
                if key:
                    word_index.setdefault(key, []).append(w)

        # Build a cell lookup: normalised cell text → Cell
        cell_index: Dict[str, List[Any]] = {}
        for tbl in tables:
            for cell in tbl.cells:
                key = cell.text.strip().lower()
                if key:
                    cell_index.setdefault(key, []).append(cell)

        def _normalise(v: Any) -> str:
            return str(v).strip().lower()

        def _merge_bboxes(boxes: List[BoundingBox]) -> BoundingBox:
            return BoundingBox(
                x1=min(b.x1 for b in boxes),
                y1=min(b.y1 for b in boxes),
                x2=max(b.x2 for b in boxes),
                y2=max(b.y2 for b in boxes),
            )

        def _find_bbox(value: Any) -> Optional[BoundingBox]:
            """Try each strategy in priority order."""
            norm = _normalise(value)
            if not norm or norm in ("n/a", "none", "null", ""):
                return None

            # 1) Exact word match
            if norm in word_index:
                best = max(word_index[norm], key=lambda w: w.confidence)
                return best.bbox

            # 2) Exact line match
            for tl in text_lines:
                if tl.text.strip().lower() == norm:
                    return tl.bbox

            # 3) Substring / multi-word span inside a text line
            for tl in text_lines:
                line_lower = tl.text.lower()
                if norm in line_lower and tl.words:
                    # Find which words belong to the matching span
                    start_idx = line_lower.index(norm)
                    end_idx = start_idx + len(norm)
                    span_boxes = []
                    cursor = 0
                    for w in tl.words:
                        w_start = line_lower.find(w.text.lower(), cursor)
                        if w_start == -1:
                            continue
                        w_end = w_start + len(w.text)
                        # Word overlaps with the matched span
                        if w_end > start_idx and w_start < end_idx:
                            span_boxes.append(w.bbox)
                        cursor = w_end
                    if span_boxes:
                        return _merge_bboxes(span_boxes)
                    # Fallback: use the entire line bbox
                    return tl.bbox

            # 4) Table cell match
            if norm in cell_index:
                best_cell = cell_index[norm][0]
                if best_cell.bbox:
                    return best_cell.bbox

            return None

        anchored = 0
        for field in fields:
            if field.bbox is None and field.value is not None:
                bbox = _find_bbox(field.value)
                if bbox:
                    field.bbox = bbox
                    anchored += 1
                    # Also update candidates from the same source
                    for cand in field.candidates:
                        if cand.bbox is None and cand.value == field.value:
                            cand.bbox = bbox

        if anchored:
            logger.info(f"Spatial anchoring: attached bboxes to {anchored} fields")
        return fields

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
        
        # Log any consistency issues AND attach to the first relevant field
        # so they appear in the validation summary output
        for result in consistency_results:
            if not result.passed:
                logger.warning(f"Consistency check failed: {result.message}")
            
            # Attach total_check to the total/total_amount field
            if result.name == "total_check":
                for f in fields:
                    if f.name.lower() in ("total", "total_amount"):
                        f.validators.append(result)
                        if not result.passed:
                            f.status = FieldStatus.VALIDATION_FAILED
                        break
            
            # Attach date_order to the due_date field
            elif result.name == "date_order":
                for f in fields:
                    if f.name.lower() == "due_date":
                        f.validators.append(result)
                        if not result.passed:
                            f.status = FieldStatus.VALIDATION_FAILED
                        break
        
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
        """Save document to JSON file and Markdown report."""
        output_path = Path(output_dir)
        # Route into Local/ or Azure_Cloud/ subfolder
        subfolder = "Azure_Cloud" if processing_mode == "azure" else "Local"
        output_path = output_path / subfolder
        output_path.mkdir(parents=True, exist_ok=True)
        
        stem = Path(document.metadata.filename).stem
        filename = f"{stem}.json"
        filepath = output_path / filename
        
        # Convert to dict and serialize
        data = document.model_dump(mode="json")
        
        # Add reconstruction prompt for LLM-friendly visual reconstruction
        from docvision.io.reconstruction import add_reconstruction_to_document
        data = add_reconstruction_to_document(data)
        
        with open(filepath, "w", encoding="utf-8") as f:
            if self.config.output.pretty_json:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            else:
                json.dump(data, f, ensure_ascii=False, default=str)
        
        logger.info(f"Output saved to: {filepath}")

        # Generate Markdown report alongside JSON
        if getattr(self.config, 'markdown', None) is None or self.config.markdown.enable:
            md_dir = getattr(self.config, 'markdown', None)
            md_base = md_dir.dir if md_dir else "markdown"
            save_markdown(
                data=data,
                output_dir=md_base,
                processing_mode=processing_mode,
                filename_stem=document.metadata.filename,
            )

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
            from copy import copy
            
            workers = self.config.runtime.get_workers()
            
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(self.process, path, copy(options) if options else None): path
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
