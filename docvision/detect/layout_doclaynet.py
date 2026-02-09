"""
Layout detection using DocLayNet-trained models.

Supports YOLO-based detection for document layout analysis.
Classifies regions as: header, footer, text, table, figure, logo, etc.
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
import numpy as np
from loguru import logger

from docvision.types import LayoutRegion, LayoutRegionType, BoundingBox

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("Ultralytics YOLO not available. Layout detection will use fallback.")


# DocLayNet class mapping for hantian/yolo-doclaynet model
# Maps model class IDs to LayoutRegionType enum values
DOCLAYNET_CLASSES = {
    0: LayoutRegionType.CAPTION,      # Caption
    1: LayoutRegionType.FOOTER,        # Footnote
    2: LayoutRegionType.FIGURE,        # Formula (treated as figure)
    3: LayoutRegionType.LIST,          # List-item
    4: LayoutRegionType.FOOTER,        # Page-footer
    5: LayoutRegionType.HEADER,        # Page-header
    6: LayoutRegionType.FIGURE,        # Picture
    7: LayoutRegionType.TITLE,         # Section-header
    8: LayoutRegionType.TABLE,         # Table
    9: LayoutRegionType.TEXT,          # Text
    10: LayoutRegionType.TITLE,        # Title
}


class LayoutDetector:
    """
    Detect document layout regions using YOLO trained on DocLayNet.
    
    Identifies semantic regions like headers, tables, text blocks, etc.
    for downstream processing.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        confidence_threshold: float = 0.5
    ):
        """
        Initialize layout detector.
        
        Args:
            model_path: Path to YOLO model weights
            device: Device to run inference on ('cpu' or 'cuda')
            confidence_threshold: Minimum confidence for detections
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.model_loaded = False
        
        if YOLO_AVAILABLE and model_path:
            self._load_model(model_path)
    
    def _load_model(self, model_path: str) -> None:
        """Load YOLO model from path."""
        path = Path(model_path)
        
        if path.exists():
            try:
                self.model = YOLO(str(path))
                self.model.to(self.device)
                self.model_loaded = True
                logger.info(f"Loaded layout model from: {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load layout model: {e}")
        else:
            logger.warning(f"Layout model not found: {model_path}")
    
    def detect(self, image: np.ndarray) -> List[LayoutRegion]:
        """
        Detect layout regions in document image.
        
        Args:
            image: Document image (BGR format)
            
        Returns:
            List of detected layout regions
        """
        if self.model_loaded and self.model is not None:
            return self._detect_with_yolo(image)
        else:
            return self._detect_fallback(image)
    
    def _detect_with_yolo(self, image: np.ndarray) -> List[LayoutRegion]:
        """Detect using YOLO model."""
        results = self.model(image, verbose=False)
        
        regions = []
        
        for result in results:
            boxes = result.boxes
            
            if boxes is None:
                continue
            
            for i in range(len(boxes)):
                # Get box coordinates
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                
                if conf < self.confidence_threshold:
                    continue
                
                # Map class to region type
                region_type = DOCLAYNET_CLASSES.get(cls_id, LayoutRegionType.UNKNOWN)
                
                region = LayoutRegion(
                    type=region_type,
                    bbox=BoundingBox(
                        x1=float(xyxy[0]),
                        y1=float(xyxy[1]),
                        x2=float(xyxy[2]),
                        y2=float(xyxy[3])
                    ),
                    confidence=conf
                )
                regions.append(region)
        
        logger.debug(f"YOLO detected {len(regions)} layout regions")
        return regions
    
    def _detect_fallback(self, image: np.ndarray) -> List[LayoutRegion]:
        """
        Fallback detection using contour analysis.
        
        Less accurate than YOLO but provides basic layout detection
        without requiring model weights.
        """
        if not CV2_AVAILABLE:
            return []
        
        h, w = image.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Dilate to connect text regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))
        dilated = cv2.dilate(binary, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        regions = []
        min_area = (h * w) * 0.001  # Minimum 0.1% of image area
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < min_area:
                continue
            
            x, y, cw, ch = cv2.boundingRect(contour)
            
            # Heuristic classification based on position and aspect ratio
            region_type = self._classify_region_heuristic(
                x, y, cw, ch, w, h
            )
            
            region = LayoutRegion(
                type=region_type,
                bbox=BoundingBox(
                    x1=float(x),
                    y1=float(y),
                    x2=float(x + cw),
                    y2=float(y + ch)
                ),
                confidence=0.5  # Lower confidence for heuristic
            )
            regions.append(region)
        
        # Merge overlapping regions
        regions = self._merge_overlapping(regions)
        
        logger.debug(f"Fallback detected {len(regions)} layout regions")
        return regions
    
    def _classify_region_heuristic(
        self,
        x: int, y: int, w: int, h: int,
        img_w: int, img_h: int
    ) -> LayoutRegionType:
        """
        Classify region based on position and size heuristics.
        
        Args:
            x, y, w, h: Region bounding box
            img_w, img_h: Image dimensions
            
        Returns:
            Inferred region type
        """
        # Calculate relative positions
        rel_y = y / img_h
        rel_h = h / img_h
        aspect_ratio = w / h if h > 0 else 0
        
        # Header: top 15% of page
        if rel_y < 0.15:
            return LayoutRegionType.HEADER
        
        # Footer: bottom 15% of page
        if rel_y + rel_h > 0.85:
            return LayoutRegionType.FOOTER
        
        # Page number: small region at bottom corners
        if rel_y + rel_h > 0.9 and (x < img_w * 0.2 or x > img_w * 0.8):
            return LayoutRegionType.PAGE_NUMBER
        
        # Table: typically wide with moderate height
        if aspect_ratio > 2.0 and rel_h > 0.1:
            return LayoutRegionType.TABLE
        
        # Title: wide, near top, short height
        if rel_y < 0.3 and aspect_ratio > 3.0 and rel_h < 0.1:
            return LayoutRegionType.TITLE
        
        # Default to text
        return LayoutRegionType.TEXT
    
    def _merge_overlapping(
        self,
        regions: List[LayoutRegion],
        iou_threshold: float = 0.5
    ) -> List[LayoutRegion]:
        """
        Merge overlapping regions of the same type.
        
        Args:
            regions: List of regions
            iou_threshold: IoU threshold for merging
            
        Returns:
            Merged regions
        """
        if len(regions) <= 1:
            return regions
        
        # Sort by area (largest first)
        regions = sorted(regions, key=lambda r: r.bbox.area, reverse=True)
        
        merged = []
        used = set()
        
        for i, region_i in enumerate(regions):
            if i in used:
                continue
            
            current_bbox = region_i.bbox
            
            for j, region_j in enumerate(regions[i+1:], start=i+1):
                if j in used:
                    continue
                
                if region_i.type != region_j.type:
                    continue
                
                # Calculate IoU
                iou = self._calculate_iou(current_bbox, region_j.bbox)
                
                if iou > iou_threshold:
                    # Merge by expanding bounding box
                    current_bbox = BoundingBox(
                        x1=min(current_bbox.x1, region_j.bbox.x1),
                        y1=min(current_bbox.y1, region_j.bbox.y1),
                        x2=max(current_bbox.x2, region_j.bbox.x2),
                        y2=max(current_bbox.y2, region_j.bbox.y2)
                    )
                    used.add(j)
            
            merged_region = LayoutRegion(
                type=region_i.type,
                bbox=current_bbox,
                confidence=region_i.confidence
            )
            merged.append(merged_region)
        
        return merged
    
    def _calculate_iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """Calculate Intersection over Union of two bounding boxes."""
        x1 = max(box1.x1, box2.x1)
        y1 = max(box1.y1, box2.y1)
        x2 = min(box1.x2, box2.x2)
        y2 = min(box1.y2, box2.y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = box1.area + box2.area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def filter_by_type(
        self,
        regions: List[LayoutRegion],
        types: List[LayoutRegionType]
    ) -> List[LayoutRegion]:
        """Filter regions by specific types."""
        return [r for r in regions if r.type in types]
    
    def get_tables(self, regions: List[LayoutRegion]) -> List[LayoutRegion]:
        """Get only table regions."""
        return self.filter_by_type(regions, [LayoutRegionType.TABLE])
    
    def get_text_regions(self, regions: List[LayoutRegion]) -> List[LayoutRegion]:
        """Get text-containing regions."""
        return self.filter_by_type(regions, [
            LayoutRegionType.TEXT,
            LayoutRegionType.HEADER,
            LayoutRegionType.FOOTER,
            LayoutRegionType.TITLE,
            LayoutRegionType.CAPTION,
            LayoutRegionType.LIST
        ])
