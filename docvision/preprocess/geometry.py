"""
Geometric preprocessing for document images.

Includes:
- Page contour detection
- Perspective correction (dewarp)
- Deskew via Hough transform
- Rotation angle detection
"""

from typing import Optional, Tuple, List
import numpy as np
from loguru import logger

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def detect_page_contour(
    image: np.ndarray,
    min_area_ratio: float = 0.3
) -> Optional[np.ndarray]:
    """
    Detect the main page/document contour in an image.
    
    Useful for photos of documents that need cropping/perspective correction.
    
    Args:
        image: Input image (BGR)
        min_area_ratio: Minimum contour area as ratio of image area
        
    Returns:
        4-point contour array, or None if not found
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV is required for geometric preprocessing")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate to connect edge segments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Find largest contour that could be a document
    image_area = image.shape[0] * image.shape[1]
    min_area = image_area * min_area_ratio
    
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        
        if area < min_area:
            continue
        
        # Approximate contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        # Check if it's a quadrilateral
        if len(approx) == 4:
            logger.debug(f"Found document contour with area ratio: {area/image_area:.2f}")
            return approx.reshape(4, 2)
    
    return None


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order points in clockwise order: top-left, top-right, bottom-right, bottom-left.
    
    Args:
        pts: 4 points as (4, 2) array
        
    Returns:
        Ordered points as (4, 2) array
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    
    # Sum of coordinates: smallest = top-left, largest = bottom-right
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Difference: smallest = top-right, largest = bottom-left
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect


def perspective_correction(
    image: np.ndarray,
    contour: Optional[np.ndarray] = None,
    padding: int = 0
) -> np.ndarray:
    """
    Apply perspective correction to straighten a document.
    
    Args:
        image: Input image (BGR)
        contour: 4-point document contour (auto-detect if None)
        padding: Padding around the corrected document
        
    Returns:
        Perspective-corrected image
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV is required for geometric preprocessing")
    
    if contour is None:
        contour = detect_page_contour(image)
        if contour is None:
            logger.warning("No document contour found, returning original image")
            return image
    
    # Order points
    pts = order_points(contour.astype(np.float32))
    tl, tr, br, bl = pts
    
    # Compute dimensions of the corrected image
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = int(max(width_top, width_bottom))
    
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = int(max(height_left, height_right))
    
    # Destination points
    dst = np.array([
        [padding, padding],
        [max_width - 1 + padding, padding],
        [max_width - 1 + padding, max_height - 1 + padding],
        [padding, max_height - 1 + padding]
    ], dtype=np.float32)
    
    # Compute perspective transform
    M = cv2.getPerspectiveTransform(pts, dst)
    
    # Apply transform
    corrected = cv2.warpPerspective(
        image, M,
        (max_width + 2 * padding, max_height + 2 * padding),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    
    logger.debug(f"Applied perspective correction: {image.shape} -> {corrected.shape}")
    return corrected


def get_rotation_angle(image: np.ndarray, max_angle: float = 15.0) -> float:
    """
    Detect rotation angle of text using Hough transform.
    
    Args:
        image: Input image (BGR or grayscale)
        max_angle: Maximum expected rotation angle
        
    Returns:
        Rotation angle in degrees (positive = counter-clockwise)
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV is required for geometric preprocessing")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Hough line detection
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=100,
        maxLineGap=10
    )
    
    if lines is None or len(lines) == 0:
        return 0.0
    
    # Calculate angles of all lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        
        # Normalize to [-max_angle, max_angle]
        if abs(angle) <= max_angle:
            angles.append(angle)
        elif abs(angle - 90) <= max_angle:
            angles.append(angle - 90)
        elif abs(angle + 90) <= max_angle:
            angles.append(angle + 90)
    
    if not angles:
        return 0.0
    
    # Use median angle to be robust against outliers
    angle = np.median(angles)
    
    logger.debug(f"Detected rotation angle: {angle:.2f}° from {len(angles)} lines")
    return angle


def deskew_image(
    image: np.ndarray,
    angle: Optional[float] = None,
    background_color: Tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    """
    Deskew an image by rotating to correct detected skew.
    
    Args:
        image: Input image (BGR)
        angle: Rotation angle in degrees (auto-detect if None)
        background_color: Color for newly exposed areas
        
    Returns:
        Deskewed image
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV is required for geometric preprocessing")
    
    if angle is None:
        angle = get_rotation_angle(image)
    
    # Skip if angle is negligible
    if abs(angle) < 0.1:
        logger.debug("Skew angle negligible, skipping deskew")
        return image
    
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Compute rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Compute new image bounds to avoid cropping
    cos_val = abs(M[0, 0])
    sin_val = abs(M[0, 1])
    new_w = int(h * sin_val + w * cos_val)
    new_h = int(h * cos_val + w * sin_val)
    
    # Adjust rotation matrix for new bounds
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    
    # Apply rotation
    rotated = cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=background_color
    )
    
    logger.debug(f"Applied deskew: {angle:.2f}°, {image.shape} -> {rotated.shape}")
    return rotated


def crop_to_content(
    image: np.ndarray,
    padding: int = 10,
    background_threshold: int = 250
) -> np.ndarray:
    """
    Crop image to content area, removing excessive white borders.
    
    Args:
        image: Input image (BGR)
        padding: Padding around content
        background_threshold: Grayscale value above which is considered background
        
    Returns:
        Cropped image
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV is required for geometric preprocessing")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold to find content
    _, thresh = cv2.threshold(gray, background_threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Find non-zero pixels
    coords = cv2.findNonZero(thresh)
    
    if coords is None:
        return image
    
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(coords)
    
    # Add padding
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image.shape[1], x + w + padding)
    y2 = min(image.shape[0], y + h + padding)
    
    cropped = image[y1:y2, x1:x2]
    
    logger.debug(f"Cropped to content: {image.shape} -> {cropped.shape}")
    return cropped
