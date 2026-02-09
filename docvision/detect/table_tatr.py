"""
Table structure detection using TATR (Table Transformer).

Detects table rows, columns, and cells including borderless tables.
Uses Microsoft's Table Transformer for structure recognition.
"""

from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import numpy as np
from loguru import logger

from docvision.types import Table, Cell, BoundingBox

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import torch
    from transformers import (
        TableTransformerForObjectDetection,
        DetrImageProcessor
    )
    TATR_AVAILABLE = True
except ImportError:
    TATR_AVAILABLE = False
    logger.warning("Table Transformer not available. Table detection will use fallback.")


class TableDetector:
    """
    Detect and analyze table structure in documents.
    
    Uses Microsoft's Table Transformer (TATR) for accurate detection
    of table cells, rows, and columns, including borderless tables.
    """
    
    def __init__(
        self,
        model_name: str = "models/table-transformer-structure",
        device: str = "cpu",
        confidence_threshold: float = 0.5
    ):
        """
        Initialize table detector.
        
        Args:
            model_name: HuggingFace model name or local path
            device: Device for inference ('cpu' or 'cuda')
            confidence_threshold: Minimum confidence for detections
        """
        self.model_name = model_name
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.processor = None
        self.model_loaded = False
        
        if TATR_AVAILABLE:
            self._load_model()
    
    def _load_model(self) -> None:
        """Load Table Transformer model."""
        try:
            self.processor = DetrImageProcessor.from_pretrained(self.model_name)
            self.model = TableTransformerForObjectDetection.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            logger.info(f"Loaded Table Transformer: {self.model_name}")
        except Exception as e:
            logger.warning(f"Failed to load Table Transformer: {e}")
    
    def detect(
        self,
        image: np.ndarray,
        table_bbox: Optional[BoundingBox] = None
    ) -> List[Table]:
        """
        Detect tables and their structure in image.
        
        Args:
            image: Document image (BGR format)
            table_bbox: If provided, analyze only this region
            
        Returns:
            List of detected tables with cell structure
        """
        if table_bbox:
            # Crop to table region
            x1, y1 = int(table_bbox.x1), int(table_bbox.y1)
            x2, y2 = int(table_bbox.x2), int(table_bbox.y2)
            table_image = image[y1:y2, x1:x2]
            offset = (x1, y1)
        else:
            table_image = image
            offset = (0, 0)
        
        if self.model_loaded:
            tables = self._detect_with_tatr(table_image, offset)
        else:
            tables = self._detect_fallback(table_image, offset)
        
        return tables
    
    def _detect_with_tatr(
        self,
        image: np.ndarray,
        offset: Tuple[int, int] = (0, 0)
    ) -> List[Table]:
        """Detect using Table Transformer."""
        import torch
        from PIL import Image
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Process image
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Process results
        target_sizes = torch.tensor([[pil_image.height, pil_image.width]])
        results = self.processor.post_process_object_detection(
            outputs, 
            threshold=self.confidence_threshold,
            target_sizes=target_sizes
        )[0]
        
        # Parse detections
        rows = []
        cols = []
        cells = []
        spanning_cells = []
        
        labels = results["labels"].cpu().numpy()
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        
        # TATR class mapping
        # 0: table, 1: table column, 2: table row, 3: table column header,
        # 4: table projected row header, 5: table spanning cell
        
        for label, box, score in zip(labels, boxes, scores):
            x1, y1, x2, y2 = box
            
            # Apply offset
            x1 += offset[0]
            y1 += offset[1]
            x2 += offset[0]
            y2 += offset[1]
            
            bbox = BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))
            
            if label == 1:  # Column
                cols.append({"bbox": bbox, "score": float(score)})
            elif label == 2:  # Row
                rows.append({"bbox": bbox, "score": float(score)})
            elif label == 5:  # Spanning cell
                spanning_cells.append({"bbox": bbox, "score": float(score)})
        
        # Build table structure
        tables = self._build_table_structure(rows, cols, spanning_cells, image.shape, offset)
        
        return tables
    
    def _detect_fallback(
        self,
        image: np.ndarray,
        offset: Tuple[int, int] = (0, 0)
    ) -> List[Table]:
        """
        Fallback table detection using line detection.
        
        Detects tables based on grid lines or aligned text.
        """
        if not CV2_AVAILABLE:
            return []
        
        h, w = image.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect horizontal and vertical lines
        horizontal_lines = self._detect_lines(gray, "horizontal")
        vertical_lines = self._detect_lines(gray, "vertical")
        
        # If we have grid lines, extract table structure
        if len(horizontal_lines) >= 2 and len(vertical_lines) >= 2:
            tables = self._build_from_lines(
                horizontal_lines, vertical_lines, h, w, offset
            )
            if tables:
                return tables
        
        # Fallback: detect table based on text alignment
        return self._detect_from_text_alignment(image, offset)
    
    def _detect_lines(
        self,
        gray: np.ndarray,
        direction: str
    ) -> List[Tuple[int, int, int, int]]:
        """Detect horizontal or vertical lines."""
        h, w = gray.shape
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Create structural element based on direction
        if direction == "horizontal":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 10, 1))
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 10))
        
        # Morphological operations to isolate lines
        mask = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
        
        # Find lines using Hough transform
        lines = cv2.HoughLinesP(
            mask,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=w // 5 if direction == "horizontal" else h // 5,
            maxLineGap=10
        )
        
        if lines is None:
            return []
        
        # Convert to list of tuples
        result = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            result.append((x1, y1, x2, y2))
        
        return result
    
    def _build_from_lines(
        self,
        h_lines: List[Tuple[int, int, int, int]],
        v_lines: List[Tuple[int, int, int, int]],
        img_h: int,
        img_w: int,
        offset: Tuple[int, int]
    ) -> List[Table]:
        """Build table structure from detected lines."""
        # Get unique row positions (y-coordinates of horizontal lines)
        row_positions = sorted(set(
            [(l[1] + l[3]) // 2 for l in h_lines]
        ))
        
        # Get unique column positions (x-coordinates of vertical lines)
        col_positions = sorted(set(
            [(l[0] + l[2]) // 2 for l in v_lines]
        ))
        
        if len(row_positions) < 2 or len(col_positions) < 2:
            return []
        
        # Build cells from grid
        cells = []
        num_rows = len(row_positions) - 1
        num_cols = len(col_positions) - 1
        
        for r in range(num_rows):
            for c in range(num_cols):
                cell = Cell(
                    row=r,
                    col=c,
                    bbox=BoundingBox(
                        x1=float(col_positions[c] + offset[0]),
                        y1=float(row_positions[r] + offset[1]),
                        x2=float(col_positions[c + 1] + offset[0]),
                        y2=float(row_positions[r + 1] + offset[1])
                    ),
                    is_header=(r == 0)  # Assume first row is header
                )
                cells.append(cell)
        
        table = Table(
            page=1,
            bbox=BoundingBox(
                x1=float(col_positions[0] + offset[0]),
                y1=float(row_positions[0] + offset[1]),
                x2=float(col_positions[-1] + offset[0]),
                y2=float(row_positions[-1] + offset[1])
            ),
            rows=num_rows,
            cols=num_cols,
            cells=cells,
            confidence=0.7,
            has_borders=True
        )
        
        return [table]
    
    def _detect_from_text_alignment(
        self,
        image: np.ndarray,
        offset: Tuple[int, int]
    ) -> List[Table]:
        """
        Detect borderless tables from text alignment patterns.
        
        Analyzes horizontal and vertical text alignment to identify
        potential table structures.
        """
        # This is a simplified implementation
        # A full implementation would analyze text positions for alignment
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find connected components (text blobs)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        
        if num_labels < 5:  # Not enough text elements
            return []
        
        # Analyze horizontal alignment (potential rows)
        y_positions = []
        for i in range(1, num_labels):  # Skip background
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 50:  # Skip noise
                continue
            y = centroids[i][1]
            y_positions.append((y, i))
        
        if len(y_positions) < 4:
            return []
        
        # Cluster y-positions to find rows
        y_positions.sort()
        rows = self._cluster_positions([y for y, _ in y_positions], threshold=20)
        
        if len(rows) < 2:
            return []
        
        # Similarly analyze x-positions for columns
        x_positions = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 50:
                continue
            x = stats[i, cv2.CC_STAT_LEFT]
            x_positions.append(x)
        
        cols = self._cluster_positions(sorted(x_positions), threshold=30)
        
        if len(cols) < 2:
            return []
        
        # Build table
        h, w = image.shape[:2]
        
        cells = []
        for r, row_y in enumerate(rows):
            for c, col_x in enumerate(cols):
                next_row_y = rows[r + 1] if r + 1 < len(rows) else h
                next_col_x = cols[c + 1] if c + 1 < len(cols) else w
                
                cell = Cell(
                    row=r,
                    col=c,
                    bbox=BoundingBox(
                        x1=float(col_x + offset[0]),
                        y1=float(row_y + offset[1]),
                        x2=float(next_col_x + offset[0]),
                        y2=float(next_row_y + offset[1])
                    ),
                    is_header=(r == 0)
                )
                cells.append(cell)
        
        table = Table(
            page=1,
            bbox=BoundingBox(
                x1=float(cols[0] + offset[0]),
                y1=float(rows[0] + offset[1]),
                x2=float(cols[-1] + offset[0]),
                y2=float(rows[-1] + offset[1])
            ),
            rows=len(rows),
            cols=len(cols),
            cells=cells,
            confidence=0.5,  # Lower confidence for borderless detection
            has_borders=False
        )
        
        return [table]
    
    def _cluster_positions(
        self,
        positions: List[float],
        threshold: float
    ) -> List[float]:
        """Cluster nearby positions into groups."""
        if not positions:
            return []
        
        clusters = []
        current_cluster = [positions[0]]
        
        for pos in positions[1:]:
            if pos - current_cluster[-1] <= threshold:
                current_cluster.append(pos)
            else:
                clusters.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [pos]
        
        clusters.append(sum(current_cluster) / len(current_cluster))
        return clusters
    
    def _build_table_structure(
        self,
        rows: List[Dict],
        cols: List[Dict],
        spanning_cells: List[Dict],
        img_shape: Tuple[int, ...],
        offset: Tuple[int, int]
    ) -> List[Table]:
        """Build table structure from TATR detections."""
        if not rows or not cols:
            return []
        
        # Sort rows and columns by position
        rows = sorted(rows, key=lambda r: r["bbox"].y1)
        cols = sorted(cols, key=lambda c: c["bbox"].x1)
        
        num_rows = len(rows)
        num_cols = len(cols)
        
        # Build cells by intersecting rows and columns
        cells = []
        
        for r_idx, row in enumerate(rows):
            for c_idx, col in enumerate(cols):
                # Cell bbox is intersection of row and column
                x1 = col["bbox"].x1
                x2 = col["bbox"].x2
                y1 = row["bbox"].y1
                y2 = row["bbox"].y2
                
                cell = Cell(
                    row=r_idx,
                    col=c_idx,
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    confidence=(row["score"] + col["score"]) / 2,
                    is_header=(r_idx == 0)
                )
                cells.append(cell)
        
        # Calculate overall table bbox
        table_bbox = BoundingBox(
            x1=min(c["bbox"].x1 for c in cols),
            y1=min(r["bbox"].y1 for r in rows),
            x2=max(c["bbox"].x2 for c in cols),
            y2=max(r["bbox"].y2 for r in rows)
        )
        
        avg_confidence = (
            sum(r["score"] for r in rows) / len(rows) +
            sum(c["score"] for c in cols) / len(cols)
        ) / 2
        
        table = Table(
            page=1,
            bbox=table_bbox,
            rows=num_rows,
            cols=num_cols,
            cells=cells,
            confidence=avg_confidence,
            has_borders=True
        )
        
        return [table]
