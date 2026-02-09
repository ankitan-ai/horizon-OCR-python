"""
Image cropping utilities for OCR.

Provides functions for extracting text regions from images
with proper padding and normalization.
"""

from typing import List, Tuple, Optional, Union
import numpy as np
from loguru import logger

from docvision.types import BoundingBox, Polygon, TextLine

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def crop_text_region(
    image: np.ndarray,
    bbox: BoundingBox,
    padding: int = 2,
    min_size: int = 10
) -> np.ndarray:
    """
    Crop a text region from image.
    
    Args:
        image: Source image (BGR)
        bbox: Bounding box of region
        padding: Padding around region
        min_size: Minimum dimension of output
        
    Returns:
        Cropped image region
    """
    h, w = image.shape[:2]
    
    # Apply padding with bounds checking
    x1 = max(0, int(bbox.x1) - padding)
    y1 = max(0, int(bbox.y1) - padding)
    x2 = min(w, int(bbox.x2) + padding)
    y2 = min(h, int(bbox.y2) + padding)
    
    # Ensure minimum size
    if x2 - x1 < min_size:
        center_x = (x1 + x2) // 2
        x1 = max(0, center_x - min_size // 2)
        x2 = min(w, x1 + min_size)
    
    if y2 - y1 < min_size:
        center_y = (y1 + y2) // 2
        y1 = max(0, center_y - min_size // 2)
        y2 = min(h, y1 + min_size)
    
    crop = image[y1:y2, x1:x2]
    
    return crop


def crop_polygon_region(
    image: np.ndarray,
    polygon: Polygon,
    padding: int = 2
) -> np.ndarray:
    """
    Crop a polygon region with perspective correction.
    
    For quadrilateral polygons, applies perspective transform
    to rectify the region. For other shapes, uses bounding box.
    
    Args:
        image: Source image (BGR)
        polygon: Polygon defining region
        padding: Padding around region
        
    Returns:
        Cropped and rectified image
    """
    if not CV2_AVAILABLE:
        return crop_text_region(image, polygon.bounding_box, padding)
    
    if len(polygon.points) != 4:
        # Non-quadrilateral: use bounding box
        return crop_text_region(image, polygon.bounding_box, padding)
    
    # Order points: top-left, top-right, bottom-right, bottom-left
    pts = np.array(polygon.points, dtype=np.float32)
    
    # Sort by y-coordinate
    pts = pts[np.argsort(pts[:, 1])]
    top = pts[:2]
    bottom = pts[2:]
    
    # Sort top by x, bottom by x
    top = top[np.argsort(top[:, 0])]
    bottom = bottom[np.argsort(bottom[:, 0])]
    
    ordered = np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)
    
    # Calculate output dimensions
    width = int(max(
        np.linalg.norm(ordered[0] - ordered[1]),
        np.linalg.norm(ordered[3] - ordered[2])
    )) + 2 * padding
    
    height = int(max(
        np.linalg.norm(ordered[0] - ordered[3]),
        np.linalg.norm(ordered[1] - ordered[2])
    )) + 2 * padding
    
    # Destination points
    dst = np.array([
        [padding, padding],
        [width - padding - 1, padding],
        [width - padding - 1, height - padding - 1],
        [padding, height - padding - 1]
    ], dtype=np.float32)
    
    # Perspective transform
    M = cv2.getPerspectiveTransform(ordered, dst)
    cropped = cv2.warpPerspective(
        image, M, (width, height),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    
    return cropped


def normalize_crop(
    crop: np.ndarray,
    target_height: int = 32,
    maintain_aspect: bool = True,
    max_width: int = 800
) -> np.ndarray:
    """
    Normalize crop to standard size for OCR.
    
    Args:
        crop: Input image crop
        target_height: Target height in pixels
        maintain_aspect: Whether to maintain aspect ratio
        max_width: Maximum width (crop if exceeded)
        
    Returns:
        Normalized image
    """
    if not CV2_AVAILABLE:
        return crop
    
    h, w = crop.shape[:2]
    
    if h == 0 or w == 0:
        return crop
    
    if maintain_aspect:
        # Scale to target height
        scale = target_height / h
        new_h = target_height
        new_w = int(w * scale)
        
        # Limit width
        if new_w > max_width:
            new_w = max_width
    else:
        new_h = target_height
        new_w = min(int(w * (target_height / h)), max_width)
    
    # Resize
    normalized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    return normalized


def batch_crop_regions(
    image: np.ndarray,
    regions: List[Union[BoundingBox, Polygon, TextLine]],
    padding: int = 2,
    normalize: bool = False,
    target_height: int = 32
) -> List[np.ndarray]:
    """
    Batch crop multiple regions from image.
    
    Args:
        image: Source image
        regions: List of regions (BoundingBox, Polygon, or TextLine)
        padding: Padding around each region
        normalize: Whether to normalize crops
        target_height: Target height if normalizing
        
    Returns:
        List of cropped images
    """
    crops = []
    
    for region in regions:
        # Extract bbox or polygon
        if isinstance(region, TextLine):
            if region.polygon:
                crop = crop_polygon_region(image, region.polygon, padding)
            else:
                crop = crop_text_region(image, region.bbox, padding)
        elif isinstance(region, Polygon):
            crop = crop_polygon_region(image, region, padding)
        elif isinstance(region, BoundingBox):
            crop = crop_text_region(image, region, padding)
        else:
            logger.warning(f"Unknown region type: {type(region)}")
            continue
        
        if normalize:
            crop = normalize_crop(crop, target_height)
        
        crops.append(crop)
    
    return crops


def pad_to_square(image: np.ndarray, fill_value: int = 255) -> np.ndarray:
    """
    Pad image to square with white background.
    
    Args:
        image: Input image
        fill_value: Padding fill value (default white)
        
    Returns:
        Square padded image
    """
    if not CV2_AVAILABLE:
        return image
    
    h, w = image.shape[:2]
    
    if h == w:
        return image
    
    size = max(h, w)
    
    if len(image.shape) == 3:
        result = np.full((size, size, image.shape[2]), fill_value, dtype=image.dtype)
    else:
        result = np.full((size, size), fill_value, dtype=image.dtype)
    
    # Center the image
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    
    result[y_offset:y_offset+h, x_offset:x_offset+w] = image
    
    return result


def extract_table_cell_crops(
    image: np.ndarray,
    cells: List["Cell"],
    padding: int = 2
) -> List[Tuple[np.ndarray, "Cell"]]:
    """
    Extract crops for table cells.
    
    Args:
        image: Source image
        cells: List of table cells
        padding: Padding around each cell
        
    Returns:
        List of (crop, cell) tuples
    """
    from docvision.types import Cell
    
    results = []
    
    for cell in cells:
        if cell.bbox is None:
            continue
        
        crop = crop_text_region(image, cell.bbox, padding)
        
        # Skip very small crops
        if crop.shape[0] < 5 or crop.shape[1] < 5:
            continue
        
        results.append((crop, cell))
    
    return results


def mask_outside_region(
    image: np.ndarray,
    bbox: BoundingBox,
    mask_value: int = 255
) -> np.ndarray:
    """
    Mask everything outside a bounding box.
    
    Useful for focusing OCR on a specific region.
    
    Args:
        image: Input image
        bbox: Region to keep
        mask_value: Value for masked areas
        
    Returns:
        Masked image
    """
    if not CV2_AVAILABLE:
        return image
    
    result = np.full_like(image, mask_value)
    
    x1, y1 = int(bbox.x1), int(bbox.y1)
    x2, y2 = int(bbox.x2), int(bbox.y2)
    
    # Clamp to image bounds
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    result[y1:y2, x1:x2] = image[y1:y2, x1:x2]
    
    return result
