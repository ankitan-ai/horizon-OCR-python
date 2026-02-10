"""
Azure Document Intelligence provider for DocVision.

Sends page images to Azure's prebuilt-layout (or prebuilt-read / prebuilt-invoice)
model and maps the response into DocVision's native types:
    - paragraphs / lines / words  →  List[TextLine]
    - tables                      →  List[Table]
    - paragraphs with roles       →  List[LayoutRegion]

This single API call replaces five local models:
    YOLO layout, CRAFT text detection, TrOCR, Table Transformer, Tesseract.
"""

from __future__ import annotations

import io
import time
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
from loguru import logger

from docvision.config import AzureConfig
from docvision.types import (
    BoundingBox,
    Polygon,
    Word,
    TextLine,
    LayoutRegion,
    LayoutRegionType,
    Table,
    Cell,
    ContentType,
    SourceEngine,
)


# ── Azure paragraph role → DocVision LayoutRegionType mapping ───────────────
_ROLE_MAP: Dict[str, LayoutRegionType] = {
    "title": LayoutRegionType.TITLE,
    "sectionHeading": LayoutRegionType.TITLE,
    "pageHeader": LayoutRegionType.HEADER,
    "pageFooter": LayoutRegionType.FOOTER,
    "pageNumber": LayoutRegionType.PAGE_NUMBER,
    "footnote": LayoutRegionType.FOOTER,
}


class AzureDocIntelligenceProvider:
    """
    Cloud-based OCR / layout / table extraction via Azure Document Intelligence.

    Usage::

        provider = AzureDocIntelligenceProvider(config.azure)
        result = provider.analyze(page_image)
        text_lines = result["text_lines"]
        tables     = result["tables"]
        regions    = result["layout_regions"]
        raw_text   = result["raw_text"]
    """

    def __init__(
        self,
        azure_config: AzureConfig,
        cost_tracker=None,
        response_cache=None,
    ) -> None:
        if not azure_config.is_azure_ready:
            raise ValueError(
                "Azure Document Intelligence is not configured. "
                "Set AZURE_DOC_INTELLIGENCE_ENDPOINT and AZURE_DOC_INTELLIGENCE_KEY "
                "in your .env file or config YAML."
            )

        self._config = azure_config
        self._client = None  # lazy
        self.cost_tracker = cost_tracker
        self.cache = response_cache
        logger.info(
            "AzureDocIntelligenceProvider initialised  "
            f"(model={azure_config.doc_intelligence_model}, "
            f"endpoint=…{azure_config.doc_intelligence_endpoint[-30:]})"
        )

    # ── lazy client ──────────────────────────────────────────────────────────

    @property
    def client(self):
        """Lazy-initialise the SDK client (avoids import cost at startup)."""
        if self._client is None:
            from azure.ai.documentintelligence import DocumentIntelligenceClient
            from azure.core.credentials import AzureKeyCredential
            import certifi

            self._client = DocumentIntelligenceClient(
                endpoint=self._config.doc_intelligence_endpoint,
                credential=AzureKeyCredential(self._config.doc_intelligence_key),
                connection_verify=certifi.where(),
            )
            logger.debug("Azure DocumentIntelligenceClient created")
        return self._client

    # ── public API ───────────────────────────────────────────────────────────

    def analyze(
        self,
        image: np.ndarray,
        page_num: int = 1,
    ) -> Dict[str, Any]:
        """
        Send a single page image to Azure and return DocVision-native structures.

        Args:
            image: BGR or RGB numpy array (H×W×3).
            page_num: 1-based page number (for Table.page assignment).

        Returns:
            dict with keys:
                ``text_lines``     – ``List[TextLine]``
                ``tables``         – ``List[Table]``
                ``layout_regions`` – ``List[LayoutRegion]``
                ``raw_text``       – ``str``
        """
        from azure.ai.documentintelligence.models import AnalyzeDocumentRequest

        image_bytes = self._encode_image(image)
        h, w = image.shape[:2]

        # ── Check cache ─────────────────────────────────────────────
        cache_key = None
        if self.cache is not None:
            cache_key = self.cache.make_key(
                image_bytes, service="di", model=self._config.doc_intelligence_model
            )
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.info(f"Cache hit for DI page {page_num} — skipping API call")
                if self.cost_tracker:
                    self.cost_tracker.record_di_call(
                        pages=1, model=self._config.doc_intelligence_model,
                        latency=0.0, cached=True,
                    )
                return cached

        t0 = time.perf_counter()
        logger.info(
            f"Sending page {page_num} to Azure Document Intelligence "
            f"({w}×{h}, {len(image_bytes) / 1024:.0f} KB)"
        )

        poller = self.client.begin_analyze_document(
            model_id=self._config.doc_intelligence_model,
            body=AnalyzeDocumentRequest(bytes_source=image_bytes),
            output_content_format="text",
        )
        result = poller.result()

        elapsed = time.perf_counter() - t0
        logger.info(f"Azure response received in {elapsed:.2f}s")

        # Azure returns all pages; we sent a single image → page index 0
        azure_page = result.pages[0] if result.pages else None

        text_lines = self._map_text_lines(azure_page, w, h)
        tables = self._map_tables(result.tables, page_num, w, h)
        layout_regions = self._map_layout_regions(result.paragraphs, w, h)
        raw_text = result.content or ""

        logger.info(
            f"Azure page {page_num}: "
            f"{len(text_lines)} text lines, "
            f"{len(tables)} tables, "
            f"{len(layout_regions)} layout regions"
        )

        response = {
            "text_lines": text_lines,
            "tables": tables,
            "layout_regions": layout_regions,
            "raw_text": raw_text,
        }

        # ── Record cost ─────────────────────────────────────────────
        if self.cost_tracker:
            self.cost_tracker.record_di_call(
                pages=1, model=self._config.doc_intelligence_model,
                latency=elapsed,
            )

        # ── Store in cache ──────────────────────────────────────────
        if self.cache is not None and cache_key:
            self.cache.put(cache_key, response, metadata={
                "page_num": page_num, "width": w, "height": h,
            })

        return response

    def analyze_bytes(
        self,
        file_bytes: bytes,
        page_num: int = 1,
    ) -> Dict[str, Any]:
        """
        Analyse a document from raw file bytes (PDF or image).

        Convenience wrapper when you already have the bytes and don't need
        to convert from numpy.  The response mapping is identical.
        """
        from azure.ai.documentintelligence.models import AnalyzeDocumentRequest

        # ── Check cache ─────────────────────────────────────────────
        cache_key = None
        if self.cache is not None:
            cache_key = self.cache.make_key(
                file_bytes, service="di_batch", model=self._config.doc_intelligence_model
            )
            cached = self.cache.get(cache_key)
            if cached is not None:
                num_pages = len(cached.get("pages", [cached]))
                logger.info(
                    f"Cache hit for DI batch ({num_pages} page(s)) — skipping API call"
                )
                if self.cost_tracker:
                    self.cost_tracker.record_di_call(
                        pages=num_pages,
                        model=self._config.doc_intelligence_model,
                        latency=0.0, cached=True,
                    )
                return cached

        t0 = time.perf_counter()
        logger.info(
            f"Sending {len(file_bytes) / 1024:.0f} KB to Azure Document Intelligence"
        )

        poller = self.client.begin_analyze_document(
            model_id=self._config.doc_intelligence_model,
            body=AnalyzeDocumentRequest(bytes_source=file_bytes),
            output_content_format="text",
        )
        result = poller.result()

        elapsed = time.perf_counter() - t0
        logger.info(f"Azure response received in {elapsed:.2f}s")

        # Build per-page results
        all_results = []
        for idx, azure_page in enumerate(result.pages or []):
            pw = azure_page.width or 1
            ph = azure_page.height or 1

            text_lines = self._map_text_lines(azure_page, pw, ph)
            tables = self._map_tables(
                self._tables_for_page(result.tables, idx + 1),
                idx + 1,
                pw,
                ph,
            )
            layout_regions = self._map_layout_regions(
                self._paragraphs_for_page(result.paragraphs, idx + 1),
                pw,
                ph,
            )

            all_results.append(
                {
                    "text_lines": text_lines,
                    "tables": tables,
                    "layout_regions": layout_regions,
                    "raw_text": result.content or "",
                    "page_width": pw,
                    "page_height": ph,
                }
            )

        # If caller asked for a single page, return that; else full list
        if len(all_results) == 1:
            response = all_results[0]
        else:
            response = {"pages": all_results, "raw_text": result.content or ""}

        # ── Record cost (one call, N pages) ─────────────────────────
        num_pages = len(all_results)
        if self.cost_tracker:
            self.cost_tracker.record_di_call(
                pages=num_pages,
                model=self._config.doc_intelligence_model,
                latency=elapsed,
            )

        # ── Store in cache ──────────────────────────────────────────
        if self.cache is not None and cache_key:
            self.cache.put(cache_key, response, metadata={
                "pages": num_pages,
            })

        return response

    # ── image encoding ───────────────────────────────────────────────────────

    @staticmethod
    def _encode_image(image: np.ndarray) -> bytes:
        """Encode numpy image to PNG bytes for Azure API."""
        success, buf = cv2.imencode(".png", image)
        if not success:
            raise RuntimeError("Failed to encode image to PNG")
        return buf.tobytes()

    # ── text-line mapping ────────────────────────────────────────────────────

    def _map_text_lines(
        self,
        azure_page,
        page_w: float,
        page_h: float,
    ) -> List[TextLine]:
        """Map Azure ``DocumentPage.lines`` + ``words`` → ``List[TextLine]``."""
        if azure_page is None:
            return []

        # Build word lookup by span offset for matching words to lines
        word_lookup: Dict[int, Any] = {}
        for w in azure_page.words or []:
            if w.span:
                word_lookup[w.span.offset] = w

        text_lines: List[TextLine] = []

        for line in azure_page.lines or []:
            poly = self._polygon_from_flat(line.polygon, page_w, page_h)
            bbox = poly.bounding_box if poly else BoundingBox(x1=0, y1=0, x2=1, y2=1)

            # Collect words that fall within this line's span
            words = self._words_for_line(line, azure_page.words or [], page_w, page_h)

            # Line confidence = average word confidence
            confidences = [wd.confidence for wd in words] if words else [0.9]
            avg_conf = sum(confidences) / len(confidences)

            text_lines.append(
                TextLine(
                    text=line.content or "",
                    words=words,
                    polygon=poly,
                    bbox=bbox,
                    confidence=min(max(avg_conf, 0.0), 1.0),
                    source=SourceEngine.AZURE_DOC_INTELLIGENCE,
                    content_type=ContentType.PRINTED,
                )
            )

        return text_lines

    def _words_for_line(
        self,
        line,
        all_words,
        page_w: float,
        page_h: float,
    ) -> List[Word]:
        """Find Azure ``DocumentWord`` objects that belong to a given line."""
        if not line.spans:
            return []

        line_start = line.spans[0].offset
        line_end = line_start + line.spans[0].length

        matched: List[Word] = []
        for w in all_words:
            if w.span and line_start <= w.span.offset < line_end:
                poly = self._polygon_from_flat(w.polygon, page_w, page_h)
                bbox = poly.bounding_box if poly else BoundingBox(x1=0, y1=0, x2=1, y2=1)

                matched.append(
                    Word(
                        text=w.content or "",
                        bbox=bbox,
                        confidence=min(max(w.confidence or 0.0, 0.0), 1.0),
                        source=SourceEngine.AZURE_DOC_INTELLIGENCE,
                        content_type=ContentType.PRINTED,
                    )
                )

        return matched

    # ── table mapping ────────────────────────────────────────────────────────

    def _map_tables(
        self,
        azure_tables: Optional[list],
        page_num: int,
        page_w: float,
        page_h: float,
    ) -> List[Table]:
        """Map Azure ``DocumentTable`` list → ``List[Table]``."""
        if not azure_tables:
            return []

        tables: List[Table] = []

        for at in azure_tables:
            # Table bounding box from bounding_regions
            bbox = self._bbox_from_regions(at.bounding_regions, page_w, page_h)

            cells: List[Cell] = []
            for ac in at.cells or []:
                cell_bbox = self._bbox_from_regions(
                    ac.bounding_regions, page_w, page_h
                )
                is_header = (ac.kind or "").lower() in (
                    "columnheader",
                    "rowheader",
                    "stubhead",
                )
                cells.append(
                    Cell(
                        row=ac.row_index,
                        col=ac.column_index,
                        row_span=ac.row_span or 1,
                        col_span=ac.column_span or 1,
                        text=ac.content or "",
                        bbox=cell_bbox,
                        confidence=0.95,  # Azure DI doesn't give per-cell confidence
                        source=SourceEngine.AZURE_DOC_INTELLIGENCE,
                        is_header=is_header,
                    )
                )

            tables.append(
                Table(
                    page=page_num,
                    bbox=bbox,
                    rows=at.row_count or 0,
                    cols=at.column_count or 0,
                    cells=cells,
                    confidence=0.95,
                    has_borders=True,
                )
            )

        return tables

    # ── layout-region mapping ────────────────────────────────────────────────

    def _map_layout_regions(
        self,
        paragraphs: Optional[list],
        page_w: float,
        page_h: float,
    ) -> List[LayoutRegion]:
        """Map Azure ``DocumentParagraph`` list → ``List[LayoutRegion]``."""
        if not paragraphs:
            return []

        regions: List[LayoutRegion] = []

        for para in paragraphs:
            role = (para.role or "").strip()
            region_type = _ROLE_MAP.get(role, LayoutRegionType.TEXT)
            bbox = self._bbox_from_regions(para.bounding_regions, page_w, page_h)

            regions.append(
                LayoutRegion(
                    type=region_type,
                    bbox=bbox,
                    confidence=0.95,
                    text_lines=[
                        TextLine(
                            text=para.content or "",
                            bbox=bbox,
                            confidence=0.95,
                            source=SourceEngine.AZURE_DOC_INTELLIGENCE,
                            content_type=ContentType.PRINTED,
                        )
                    ],
                    content_type=ContentType.PRINTED,
                )
            )

        return regions

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _polygon_from_flat(
        flat: Optional[list],
        page_w: float,
        page_h: float,
    ) -> Optional[Polygon]:
        """
        Convert Azure's flat [x1,y1,x2,y2,…] polygon to DocVision ``Polygon``.

        Azure DI returns coordinates in the page's unit space (inches for PDFs,
        pixels for images).  For images the page width/height already matches,
        so we keep coordinates as-is.  For PDFs (unit == "inch") we'd need to
        scale; in practice the orchestrator sends rasterised images so this is
        fine.
        """
        if not flat or len(flat) < 4:
            return None

        points = [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]
        return Polygon(points=points)

    @staticmethod
    def _bbox_from_regions(
        regions: Optional[list],
        page_w: float,
        page_h: float,
    ) -> BoundingBox:
        """Extract a BoundingBox from Azure ``bounding_regions``."""
        if not regions:
            return BoundingBox(x1=0, y1=0, x2=1, y2=1)

        # Take the first bounding region
        region = regions[0]
        poly = region.polygon
        if not poly or len(poly) < 4:
            return BoundingBox(x1=0, y1=0, x2=1, y2=1)

        xs = [poly[i] for i in range(0, len(poly), 2)]
        ys = [poly[i + 1] for i in range(0, len(poly), 2)]

        return BoundingBox(
            x1=min(xs),
            y1=min(ys),
            x2=max(xs),
            y2=max(ys),
        )

    @staticmethod
    def _tables_for_page(
        all_tables: Optional[list],
        page_number: int,
    ) -> list:
        """Filter tables that belong to a specific page."""
        if not all_tables:
            return []
        return [
            t
            for t in all_tables
            if t.bounding_regions
            and any(r.page_number == page_number for r in t.bounding_regions)
        ]

    @staticmethod
    def _paragraphs_for_page(
        all_paragraphs: Optional[list],
        page_number: int,
    ) -> list:
        """Filter paragraphs that belong to a specific page."""
        if not all_paragraphs:
            return []
        return [
            p
            for p in all_paragraphs
            if p.bounding_regions
            and any(r.page_number == page_number for r in p.bounding_regions)
        ]
