"""
Text detection using CRAFT (Character Region Awareness for Text Detection).

Detects text regions as polygons, supporting curved and arbitrary-shaped text.
Falls back to contour-based detection if CRAFT model is not available.
"""

from typing import List, Optional, Tuple
from pathlib import Path
import numpy as np
from loguru import logger

from docvision.types import TextLine, BoundingBox, Polygon, ContentType

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TextDetector:
    """
    Detect text regions in document images.
    
    Uses CRAFT model for accurate polygon detection of text lines,
    with fallback to OpenCV contour-based detection.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        text_threshold: float = 0.7,
        link_threshold: float = 0.4,
        low_text_threshold: float = 0.4,
        poly: bool = True,
        canvas_size: int = 1280,
        mag_ratio: float = 1.5,
    ):
        """
        Initialize text detector.
        
        Args:
            model_path: Path to CRAFT model weights
            device: Device for inference ('cpu' or 'cuda')
            text_threshold: Text confidence threshold
            link_threshold: Link confidence threshold
            low_text_threshold: Low text threshold for poly detection
            poly: Whether to return polygons (True) or bounding boxes (False)
            canvas_size: Maximum image dimension for CRAFT inference
            mag_ratio: Image magnification ratio for CRAFT
        """
        self.device = device
        self.text_threshold = text_threshold
        self.link_threshold = link_threshold
        self.low_text_threshold = low_text_threshold
        self.poly = poly
        self.canvas_size = canvas_size
        self.mag_ratio = mag_ratio
        self.model = None
        self.model_loaded = False
        
        if model_path and TORCH_AVAILABLE:
            self._load_model(model_path)
    
    def _load_model(self, model_path: str) -> None:
        """Load CRAFT neural network model."""
        path = Path(model_path)
        
        if not path.exists():
            logger.warning(f"CRAFT model not found: {model_path}")
            return
        
        try:
            from docvision.detect.craft_net import CRAFTNet, copy_state_dict
            
            self.model = CRAFTNet(pretrained=False, freeze=False)
            
            state_dict = torch.load(str(path), map_location=self.device, weights_only=False)
            self.model.load_state_dict(copy_state_dict(state_dict))
            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            
            logger.info(f"CRAFT model loaded from: {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load CRAFT model: {e}")
            self.model = None
            self.model_loaded = False
    
    def detect(self, image: np.ndarray) -> List[TextLine]:
        """
        Detect text lines in image.
        
        Args:
            image: Document image (BGR format)
            
        Returns:
            List of detected text lines with polygons
        """
        if self.model_loaded and self.model is not None:
            return self._detect_with_craft(image)
        else:
            return self._detect_fallback(image)
    
    def _detect_with_craft(self, image: np.ndarray) -> List[TextLine]:
        """Detect text regions using CRAFT neural network."""
        from docvision.detect.craft_utils import (
            normalize_mean_variance,
            resize_aspect_ratio,
            get_det_boxes,
            adjust_result_coordinates,
        )
        
        h, w = image.shape[:2]
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize with aspect ratio preserved
        img_resized, target_ratio, size_heatmap = resize_aspect_ratio(
            rgb_image, self.canvas_size,
            interpolation=cv2.INTER_LINEAR,
            mag_ratio=self.mag_ratio
        )
        ratio_h = ratio_w = 1 / target_ratio
        
        # Normalize
        x = normalize_mean_variance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)  # HWC -> CHW
        x = x.unsqueeze(0)  # Add batch dimension
        x = x.to(self.device)
        
        # Forward pass
        with torch.inference_mode():
            y, feature = self.model(x)
        
        # Extract score maps
        score_text = y[0, :, :, 0].cpu().numpy()
        score_link = y[0, :, :, 1].cpu().numpy()
        
        # Get detection boxes
        boxes, polys = get_det_boxes(
            score_text, score_link,
            self.text_threshold, self.link_threshold,
            self.low_text_threshold, self.poly
        )
        
        # Adjust coordinates
        boxes = adjust_result_coordinates(boxes, ratio_w, ratio_h)
        polys = adjust_result_coordinates(polys, ratio_w, ratio_h)
        
        for k in range(len(polys)):
            if polys[k] is None:
                polys[k] = boxes[k]
        
        # Convert to TextLine objects
        lines = []
        for i, (box, poly) in enumerate(zip(boxes, polys)):
            box = np.array(box)
            
            # Get bounding rectangle
            x1 = max(0, float(np.min(box[:, 0])))
            y1 = max(0, float(np.min(box[:, 1])))
            x2 = min(w, float(np.max(box[:, 0])))
            y2 = min(h, float(np.max(box[:, 1])))
            
            # Skip invalid boxes
            if x2 - x1 < 3 or y2 - y1 < 3:
                continue
            
            # Create polygon from detected points
            polygon = None
            if poly is not None:
                poly_pts = np.array(poly)
                if len(poly_pts) >= 4:
                    points = [(float(p[0]), float(p[1])) for p in poly_pts]
                    polygon = Polygon(points=points)
            
            # Compute confidence from score map region
            # Map coordinates to score map space
            sm_x1 = max(0, int(x1 / (ratio_w * 2)))
            sm_y1 = max(0, int(y1 / (ratio_h * 2)))
            sm_x2 = min(score_text.shape[1], int(x2 / (ratio_w * 2)))
            sm_y2 = min(score_text.shape[0], int(y2 / (ratio_h * 2)))
            
            if sm_x2 > sm_x1 and sm_y2 > sm_y1:
                region_scores = score_text[sm_y1:sm_y2, sm_x1:sm_x2]
                confidence = float(np.mean(region_scores[region_scores > self.low_text_threshold]))
                if np.isnan(confidence):
                    confidence = 0.5
            else:
                confidence = 0.5
            
            line = TextLine(
                text="",
                bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                polygon=polygon,
                confidence=min(confidence, 1.0)
            )
            lines.append(line)
        
        # Sort in reading order
        lines = self._sort_lines_reading_order(lines)
        
        # Merge overlapping detections into lines
        lines = self._merge_word_boxes_to_lines(lines, h)
        
        logger.info(f"CRAFT detected {len(lines)} text lines")
        return lines
    
    def _merge_word_boxes_to_lines(
        self, lines: List[TextLine], img_height: int
    ) -> List[TextLine]:
        """
        Merge nearby word-level detections into text lines.
        
        CRAFT often returns word-level boxes. This groups horizontally
        adjacent boxes on the same line.
        """
        if len(lines) <= 1:
            return lines
        
        merged = []
        used = set()
        
        for i, line in enumerate(lines):
            if i in used:
                continue
            
            current = line
            used.add(i)
            
            # Find other boxes on same line
            for j in range(i + 1, len(lines)):
                if j in used:
                    continue
                
                other = lines[j]
                
                # Check if on same line (vertical overlap > 50%)
                v_overlap = self._vertical_overlap(current.bbox, other.bbox)
                if v_overlap < 0.5:
                    continue
                
                # Check horizontal proximity
                line_height = current.bbox.y2 - current.bbox.y1
                gap = other.bbox.x1 - current.bbox.x2
                
                if gap < line_height * 2.0:  # Gap less than 2x line height
                    # Merge
                    current = TextLine(
                        text="",
                        bbox=BoundingBox(
                            x1=min(current.bbox.x1, other.bbox.x1),
                            y1=min(current.bbox.y1, other.bbox.y1),
                            x2=max(current.bbox.x2, other.bbox.x2),
                            y2=max(current.bbox.y2, other.bbox.y2),
                        ),
                        confidence=max(current.confidence, other.confidence),
                    )
                    used.add(j)
            
            merged.append(current)
        
        return merged
    
    def _detect_fallback(self, image: np.ndarray) -> List[TextLine]:
        """
        Fallback text detection using morphological operations.
        
        Creates line-level bounding boxes by analyzing text connectivity.
        """
        if not CV2_AVAILABLE:
            return []
        
        h, w = image.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 4
        )
        
        # Horizontal dilation to connect text into lines
        # Kernel width should be larger than character spacing
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
        dilated_h = cv2.dilate(binary, horizontal_kernel, iterations=1)
        
        # Slight vertical dilation to ensure full line height
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        dilated = cv2.dilate(dilated_h, vertical_kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        lines = []
        min_width = w * 0.01  # Minimum 1% of image width
        min_height = 5  # Minimum 5 pixels tall
        
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            
            # Filter by size
            if cw < min_width or ch < min_height:
                continue
            
            # Filter by aspect ratio (text lines are typically wide)
            aspect_ratio = cw / ch if ch > 0 else 0
            if aspect_ratio < 0.5:  # Too tall/narrow
                continue
            
            # Create polygon from contour for more accurate shape
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            polygon = None
            if len(approx) >= 4:
                points = [(float(p[0][0]), float(p[0][1])) for p in approx]
                polygon = Polygon(points=points)
            
            line = TextLine(
                text="",  # Will be filled by OCR
                bbox=BoundingBox(
                    x1=float(x),
                    y1=float(y),
                    x2=float(x + cw),
                    y2=float(y + ch)
                ),
                polygon=polygon,
                confidence=0.5  # Fallback confidence
            )
            lines.append(line)
        
        # Sort lines by vertical position (top to bottom, left to right)
        lines = self._sort_lines_reading_order(lines)
        
        logger.debug(f"Fallback detected {len(lines)} text lines")
        return lines
    
    def _sort_lines_reading_order(self, lines: List[TextLine]) -> List[TextLine]:
        """
        Sort text lines in reading order (top-to-bottom, left-to-right).
        
        Groups lines into rows based on vertical overlap, then sorts
        within each row by x-coordinate.
        """
        if not lines:
            return lines
        
        # Group lines by approximate row
        rows = []
        used = set()
        
        # Sort by y-coordinate first
        lines_sorted = sorted(lines, key=lambda l: l.bbox.y1)
        
        for i, line in enumerate(lines_sorted):
            if i in used:
                continue
            
            row = [line]
            used.add(i)
            
            # Find other lines in same row (vertical overlap)
            for j, other in enumerate(lines_sorted[i+1:], start=i+1):
                if j in used:
                    continue
                
                # Check vertical overlap
                overlap = self._vertical_overlap(line.bbox, other.bbox)
                if overlap > 0.5:  # At least 50% overlap
                    row.append(other)
                    used.add(j)
            
            rows.append(row)
        
        # Sort each row by x-coordinate
        sorted_lines = []
        for row in rows:
            row_sorted = sorted(row, key=lambda l: l.bbox.x1)
            sorted_lines.extend(row_sorted)
        
        return sorted_lines
    
    def _vertical_overlap(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """Calculate vertical overlap ratio between two bounding boxes."""
        y1 = max(box1.y1, box2.y1)
        y2 = min(box1.y2, box2.y2)
        
        if y2 <= y1:
            return 0.0
        
        overlap = y2 - y1
        height1 = box1.y2 - box1.y1
        height2 = box2.y2 - box2.y1
        
        return overlap / min(height1, height2) if min(height1, height2) > 0 else 0.0
    
    def detect_words(self, image: np.ndarray) -> List[Tuple[BoundingBox, float]]:
        """
        Detect individual words (finer granularity than lines).
        
        Args:
            image: Document image (BGR format)
            
        Returns:
            List of (bounding_box, confidence) tuples
        """
        if not CV2_AVAILABLE:
            return []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Minimal dilation to connect characters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        dilated = cv2.dilate(binary, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        words = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter small noise
            if w < 5 or h < 5:
                continue
            
            bbox = BoundingBox(
                x1=float(x),
                y1=float(y),
                x2=float(x + w),
                y2=float(y + h)
            )
            words.append((bbox, 0.5))
        
        # Sort by position
        words.sort(key=lambda w: (w[0].y1, w[0].x1))
        
        return words
    
    def crop_region(
        self,
        image: np.ndarray,
        bbox: BoundingBox,
        padding: int = 2
    ) -> np.ndarray:
        """
        Crop a region from the image with padding.
        
        Args:
            image: Source image
            bbox: Region bounding box
            padding: Padding around region
            
        Returns:
            Cropped image
        """
        h, w = image.shape[:2]
        
        x1 = max(0, int(bbox.x1) - padding)
        y1 = max(0, int(bbox.y1) - padding)
        x2 = min(w, int(bbox.x2) + padding)
        y2 = min(h, int(bbox.y2) + padding)
        
        return image[y1:y2, x1:x2]
    
    def crop_polygon(
        self,
        image: np.ndarray,
        polygon: Polygon,
        padding: int = 2
    ) -> np.ndarray:
        """
        Crop a polygon region, applying perspective transform if needed.
        
        Args:
            image: Source image
            polygon: Region polygon
            padding: Padding around region
            
        Returns:
            Cropped and rectified image
        """
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV required for polygon cropping")
        
        if len(polygon.points) != 4:
            # Fall back to bounding box crop
            return self.crop_region(image, polygon.bounding_box, padding)
        
        # Order points (top-left, top-right, bottom-right, bottom-left)
        pts = np.array(polygon.points, dtype=np.float32)
        
        # Simple ordering: sort by y, then x
        pts = pts[np.argsort(pts[:, 1])]
        top = pts[:2]
        bottom = pts[2:]
        top = top[np.argsort(top[:, 0])]
        bottom = bottom[np.argsort(bottom[:, 0])]
        ordered = np.array([top[0], top[1], bottom[1], bottom[0]])
        
        # Calculate destination dimensions
        width = int(max(
            np.linalg.norm(ordered[0] - ordered[1]),
            np.linalg.norm(ordered[2] - ordered[3])
        ))
        height = int(max(
            np.linalg.norm(ordered[0] - ordered[3]),
            np.linalg.norm(ordered[1] - ordered[2])
        ))
        
        # Destination points
        dst = np.array([
            [padding, padding],
            [width - 1 + padding, padding],
            [width - 1 + padding, height - 1 + padding],
            [padding, height - 1 + padding]
        ], dtype=np.float32)
        
        # Perspective transform
        M = cv2.getPerspectiveTransform(ordered, dst)
        cropped = cv2.warpPerspective(
            image, M,
            (width + 2 * padding, height + 2 * padding),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )
        
        return cropped
