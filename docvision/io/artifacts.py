"""
Artifact generation for debugging and quality analysis.

Creates visual overlays showing:
- Layout detection regions
- Text polygons and bounding boxes  
- Table structure (rows, columns, cells)
- OCR results with confidence coloring
"""

from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from loguru import logger

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from docvision.types import (
    BoundingBox, Polygon, LayoutRegion, LayoutRegionType,
    TextLine, Table, Cell, Page, Document
)


# Color scheme for layout regions (BGR format)
LAYOUT_COLORS = {
    LayoutRegionType.HEADER: (255, 100, 100),      # Light blue
    LayoutRegionType.FOOTER: (100, 100, 255),      # Light red
    LayoutRegionType.TEXT: (100, 255, 100),        # Light green
    LayoutRegionType.TABLE: (255, 255, 100),       # Cyan
    LayoutRegionType.FIGURE: (255, 100, 255),      # Magenta
    LayoutRegionType.LOGO: (100, 255, 255),        # Yellow
    LayoutRegionType.TITLE: (200, 200, 100),       # Light cyan
    LayoutRegionType.LIST: (100, 200, 200),        # Light yellow
    LayoutRegionType.CAPTION: (200, 100, 200),     # Light magenta
    LayoutRegionType.PAGE_NUMBER: (150, 150, 150), # Gray
    LayoutRegionType.SIGNATURE: (50, 150, 255),    # Orange
    LayoutRegionType.STAMP: (255, 50, 150),        # Pink
    LayoutRegionType.UNKNOWN: (128, 128, 128),     # Gray
}


def confidence_to_color(confidence: float) -> Tuple[int, int, int]:
    """
    Convert confidence score to BGR color (red=low, green=high).
    
    Args:
        confidence: Confidence score (0.0 to 1.0)
        
    Returns:
        BGR color tuple
    """
    # Red (low) -> Yellow (medium) -> Green (high)
    if confidence < 0.5:
        # Red to Yellow
        r = 255
        g = int(255 * (confidence / 0.5))
        b = 0
    else:
        # Yellow to Green
        r = int(255 * ((1.0 - confidence) / 0.5))
        g = 255
        b = 0
    
    return (b, g, r)  # BGR format


class ArtifactManager:
    """
    Manages artifact generation for document processing stages.
    
    Creates debug images showing detection and recognition results
    for quality analysis and troubleshooting.
    """
    
    def __init__(
        self,
        output_dir: str,
        enable: bool = True,
        save_layout: bool = True,
        save_text_polygons: bool = True,
        save_table_structure: bool = True,
        save_ocr_overlay: bool = True,
        save_preprocessed: bool = True
    ):
        """
        Initialize artifact manager.
        
        Args:
            output_dir: Directory to save artifacts
            enable: Enable/disable artifact generation
            save_layout: Save layout detection overlay
            save_text_polygons: Save text polygon overlay
            save_table_structure: Save table structure overlay
            save_ocr_overlay: Save OCR results overlay
            save_preprocessed: Save preprocessed images
        """
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV is required for artifact generation")
        
        self.output_dir = Path(output_dir)
        self.enable = enable
        self.save_layout = save_layout
        self.save_text_polygons = save_text_polygons
        self.save_table_structure = save_table_structure
        self._save_ocr_overlay = save_ocr_overlay
        self.save_preprocessed = save_preprocessed
        self.current_mode: Optional[str] = None  # "local" or "azure"
        
        if enable:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_document_dir(self, doc_id: str, mode: Optional[str] = None) -> Path:
        """Get artifact directory for a specific document.

        Args:
            doc_id: Document identifier.
            mode: Processing mode (``"local"`` or ``"azure"``).  Falls back
                  to ``self.current_mode`` when *None*.  When set, artifacts
                  are stored under a ``Local/`` or ``Azure_Cloud/`` subfolder.
        """
        effective_mode = mode or self.current_mode
        base = self.output_dir
        if effective_mode:
            subfolder = "Azure_Cloud" if effective_mode == "azure" else "Local"
            base = base / subfolder
        doc_dir = base / doc_id
        if self.enable:
            doc_dir.mkdir(parents=True, exist_ok=True)
        return doc_dir
    
    def save_preprocessed_image(
        self,
        image: np.ndarray,
        doc_id: str,
        page_num: int,
        stage: str = "preprocessed"
    ) -> Optional[str]:
        """
        Save preprocessed image.
        
        Args:
            image: Preprocessed image
            doc_id: Document ID
            page_num: Page number (1-indexed)
            stage: Processing stage name
            
        Returns:
            Path to saved artifact, or None if disabled
        """
        if not self.enable or not self.save_preprocessed:
            return None
        
        doc_dir = self.get_document_dir(doc_id)
        output_path = doc_dir / f"page_{page_num:03d}_{stage}.png"
        
        cv2.imwrite(str(output_path), image)
        logger.debug(f"Saved preprocessed artifact: {output_path}")
        
        return str(output_path)
    
    def save_layout_overlay(
        self,
        image: np.ndarray,
        regions: List[LayoutRegion],
        doc_id: str,
        page_num: int
    ) -> Optional[str]:
        """
        Save layout detection overlay.
        
        Args:
            image: Original image
            regions: Detected layout regions
            doc_id: Document ID
            page_num: Page number (1-indexed)
            
        Returns:
            Path to saved artifact, or None if disabled
        """
        if not self.enable or not self.save_layout:
            return None
        
        overlay = image.copy()
        
        for region in regions:
            color = LAYOUT_COLORS.get(region.type, LAYOUT_COLORS[LayoutRegionType.UNKNOWN])
            bbox = region.bbox
            
            # Draw filled rectangle with transparency
            sub_img = overlay[int(bbox.y1):int(bbox.y2), int(bbox.x1):int(bbox.x2)]
            rect = np.full(sub_img.shape, color, dtype=np.uint8)
            cv2.addWeighted(rect, 0.3, sub_img, 0.7, 0, sub_img)
            
            # Draw border
            cv2.rectangle(
                overlay,
                (int(bbox.x1), int(bbox.y1)),
                (int(bbox.x2), int(bbox.y2)),
                color,
                2
            )
            
            # Draw label
            label = f"{region.type.value} ({region.confidence:.2f})"
            cv2.putText(
                overlay,
                label,
                (int(bbox.x1), int(bbox.y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )
        
        doc_dir = self.get_document_dir(doc_id)
        output_path = doc_dir / f"page_{page_num:03d}_layout.png"
        
        cv2.imwrite(str(output_path), overlay)
        logger.debug(f"Saved layout artifact: {output_path}")
        
        return str(output_path)
    
    def save_text_polygons_overlay(
        self,
        image: np.ndarray,
        text_lines: List[TextLine],
        doc_id: str,
        page_num: int
    ) -> Optional[str]:
        """
        Save text polygon overlay.
        
        Args:
            image: Original image
            text_lines: Detected text lines with polygons
            doc_id: Document ID
            page_num: Page number (1-indexed)
            
        Returns:
            Path to saved artifact, or None if disabled
        """
        if not self.enable or not self.save_text_polygons:
            return None
        
        overlay = image.copy()
        
        for line in text_lines:
            color = confidence_to_color(line.confidence)
            
            if line.polygon and line.polygon.points:
                # Draw polygon
                pts = np.array(line.polygon.points, dtype=np.int32)
                cv2.polylines(overlay, [pts], True, color, 2)
            else:
                # Draw bounding box
                bbox = line.bbox
                cv2.rectangle(
                    overlay,
                    (int(bbox.x1), int(bbox.y1)),
                    (int(bbox.x2), int(bbox.y2)),
                    color,
                    2
                )
        
        doc_dir = self.get_document_dir(doc_id)
        output_path = doc_dir / f"page_{page_num:03d}_text_polygons.png"
        
        cv2.imwrite(str(output_path), overlay)
        logger.debug(f"Saved text polygons artifact: {output_path}")
        
        return str(output_path)
    
    def save_table_structure_overlay(
        self,
        image: np.ndarray,
        tables: List[Table],
        doc_id: str,
        page_num: int
    ) -> Optional[str]:
        """
        Save table structure overlay.
        
        Args:
            image: Original image
            tables: Detected tables with cells
            doc_id: Document ID
            page_num: Page number (1-indexed)
            
        Returns:
            Path to saved artifact, or None if disabled
        """
        if not self.enable or not self.save_table_structure:
            return None
        
        overlay = image.copy()
        
        for table_idx, table in enumerate(tables):
            # Draw table border
            bbox = table.bbox
            cv2.rectangle(
                overlay,
                (int(bbox.x1), int(bbox.y1)),
                (int(bbox.x2), int(bbox.y2)),
                (255, 255, 0),  # Cyan
                3
            )
            
            # Draw table label
            label = f"Table {table_idx + 1} ({table.rows}x{table.cols})"
            cv2.putText(
                overlay,
                label,
                (int(bbox.x1), int(bbox.y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )
            
            # Draw cells
            for cell in table.cells:
                if cell.bbox:
                    color = (0, 255, 255) if cell.is_header else (0, 200, 0)
                    cv2.rectangle(
                        overlay,
                        (int(cell.bbox.x1), int(cell.bbox.y1)),
                        (int(cell.bbox.x2), int(cell.bbox.y2)),
                        color,
                        1
                    )
                    
                    # Draw cell coordinates
                    cell_label = f"({cell.row},{cell.col})"
                    cv2.putText(
                        overlay,
                        cell_label,
                        (int(cell.bbox.x1) + 2, int(cell.bbox.y1) + 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        color,
                        1
                    )
        
        doc_dir = self.get_document_dir(doc_id)
        output_path = doc_dir / f"page_{page_num:03d}_tables.png"
        
        cv2.imwrite(str(output_path), overlay)
        logger.debug(f"Saved table structure artifact: {output_path}")
        
        return str(output_path)
    
    def save_ocr_overlay(
        self,
        image: np.ndarray,
        text_lines: List[TextLine],
        doc_id: str,
        page_num: int,
        show_text: bool = True
    ) -> Optional[str]:
        """
        Save OCR results overlay with confidence coloring.
        
        Args:
            image: Original image
            text_lines: OCR results with text and confidence
            doc_id: Document ID
            page_num: Page number (1-indexed)
            show_text: Whether to show recognized text
            
        Returns:
            Path to saved artifact, or None if disabled
        """
        if not self.enable or not self._save_ocr_overlay:
            return None
        
        overlay = image.copy()
        
        for line in text_lines:
            color = confidence_to_color(line.confidence)
            bbox = line.bbox
            
            # Draw bounding box
            cv2.rectangle(
                overlay,
                (int(bbox.x1), int(bbox.y1)),
                (int(bbox.x2), int(bbox.y2)),
                color,
                2
            )
            
            if show_text and line.text:
                # Draw text above box
                # Truncate long text
                text = line.text[:50] + "..." if len(line.text) > 50 else line.text
                label = f"{text} ({line.confidence:.2f})"
                
                # Add background for readability
                (text_w, text_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                )
                cv2.rectangle(
                    overlay,
                    (int(bbox.x1), int(bbox.y1) - text_h - 5),
                    (int(bbox.x1) + text_w, int(bbox.y1)),
                    (255, 255, 255),
                    -1
                )
                
                cv2.putText(
                    overlay,
                    label,
                    (int(bbox.x1), int(bbox.y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1
                )
        
        doc_dir = self.get_document_dir(doc_id)
        output_path = doc_dir / f"page_{page_num:03d}_ocr.png"
        
        cv2.imwrite(str(output_path), overlay)
        logger.debug(f"Saved OCR artifact: {output_path}")
        
        return str(output_path)
    
    def save_combined_overlay(
        self,
        image: np.ndarray,
        page: Page,
        doc_id: str
    ) -> Optional[str]:
        """
        Save combined overlay with all detection results.
        
        Args:
            image: Original image
            page: Page object with all extracted content
            doc_id: Document ID
            
        Returns:
            Path to saved artifact, or None if disabled
        """
        if not self.enable:
            return None
        
        overlay = image.copy()
        
        # Draw layout regions
        for region in page.layout_regions:
            color = LAYOUT_COLORS.get(region.type, LAYOUT_COLORS[LayoutRegionType.UNKNOWN])
            bbox = region.bbox
            cv2.rectangle(
                overlay,
                (int(bbox.x1), int(bbox.y1)),
                (int(bbox.x2), int(bbox.y2)),
                color,
                1
            )
        
        # Draw text lines
        for line in page.text_lines:
            color = confidence_to_color(line.confidence)
            bbox = line.bbox
            cv2.rectangle(
                overlay,
                (int(bbox.x1), int(bbox.y1)),
                (int(bbox.x2), int(bbox.y2)),
                color,
                1
            )
        
        # Draw tables
        for table in page.tables:
            bbox = table.bbox
            cv2.rectangle(
                overlay,
                (int(bbox.x1), int(bbox.y1)),
                (int(bbox.x2), int(bbox.y2)),
                (255, 255, 0),
                2
            )
        
        doc_dir = self.get_document_dir(doc_id)
        output_path = doc_dir / f"page_{page.number:03d}_combined.png"
        
        cv2.imwrite(str(output_path), overlay)
        logger.debug(f"Saved combined artifact: {output_path}")
        
        return str(output_path)
    
    def generate_summary_html(
        self,
        document: Document,
        doc_id: str
    ) -> Optional[str]:
        """
        Generate HTML summary page for artifacts.
        
        Args:
            document: Processed document
            doc_id: Document ID
            
        Returns:
            Path to HTML file, or None if disabled
        """
        if not self.enable:
            return None
        
        doc_dir = self.get_document_dir(doc_id)
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>DocVision Artifacts - {doc_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .page {{ margin-bottom: 40px; border: 1px solid #ccc; padding: 20px; }}
        .artifacts {{ display: flex; flex-wrap: wrap; gap: 10px; }}
        .artifact {{ max-width: 400px; }}
        .artifact img {{ max-width: 100%; border: 1px solid #ddd; }}
        h1, h2 {{ color: #333; }}
        .stats {{ background: #f5f5f5; padding: 10px; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <h1>DocVision Processing Artifacts</h1>
    <div class="stats">
        <strong>Document ID:</strong> {doc_id}<br>
        <strong>Pages:</strong> {document.page_count}<br>
        <strong>Fields:</strong> {len(document.fields)}<br>
        <strong>Tables:</strong> {len(document.tables)}<br>
        <strong>Validation:</strong> {"Passed" if document.validation.passed else "Failed"}
    </div>
"""
        
        for page in document.pages:
            html += f"""
    <div class="page">
        <h2>Page {page.number}</h2>
        <div class="artifacts">
"""
            # List artifact images
            for artifact_type in ["preprocessed", "layout", "text_polygons", "tables", "ocr", "combined"]:
                img_path = f"page_{page.number:03d}_{artifact_type}.png"
                if (doc_dir / img_path).exists():
                    html += f"""
            <div class="artifact">
                <h4>{artifact_type.replace("_", " ").title()}</h4>
                <img src="{img_path}" alt="{artifact_type}">
            </div>
"""
            
            html += """
        </div>
    </div>
"""
        
        html += """
</body>
</html>
"""
        
        output_path = doc_dir / "summary.html"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        
        logger.info(f"Generated artifact summary: {output_path}")
        return str(output_path)
