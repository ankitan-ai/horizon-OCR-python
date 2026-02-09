"""
Image enhancement for better OCR accuracy.

Includes:
- Non-local means denoising
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Unsharp masking (sharpening)
- Content type detection (printed vs handwritten)
"""

from typing import Tuple, Optional
from enum import Enum
import numpy as np
from loguru import logger

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from docvision.types import ContentType


def denoise_image(
    image: np.ndarray,
    strength: int = 10,
    template_window_size: int = 7,
    search_window_size: int = 21
) -> np.ndarray:
    """
    Apply Non-Local Means denoising for better OCR accuracy.
    
    NLM denoising preserves edges while removing noise, which is
    critical for maintaining text clarity.
    
    Args:
        image: Input image (BGR)
        strength: Denoising strength (higher = more denoising)
        template_window_size: Size of template patch
        search_window_size: Size of area to search for similar patches
        
    Returns:
        Denoised image
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV is required for image enhancement")
    
    # Use color denoising for BGR images
    if len(image.shape) == 3 and image.shape[2] == 3:
        denoised = cv2.fastNlMeansDenoisingColored(
            image,
            None,
            strength,  # h
            strength,  # hForColorComponents
            template_window_size,
            search_window_size
        )
    else:
        denoised = cv2.fastNlMeansDenoising(
            image,
            None,
            h=strength,
            templateWindowSize=template_window_size,
            searchWindowSize=search_window_size
        )
    
    logger.debug(f"Applied NLM denoising with strength={strength}")
    return denoised


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    CLAHE improves local contrast while limiting noise amplification,
    making it ideal for documents with varying illumination.
    
    Args:
        image: Input image (BGR or grayscale)
        clip_limit: Threshold for contrast limiting
        tile_size: Size of grid for histogram equalization
        
    Returns:
        Contrast-enhanced image
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV is required for image enhancement")
    
    # Convert to LAB color space for better results
    if len(image.shape) == 3 and image.shape[2] == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply CLAHE to L channel only
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        l_channel = clahe.apply(l_channel)
        
        # Merge and convert back to BGR
        lab = cv2.merge([l_channel, a_channel, b_channel])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        # Grayscale image
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        enhanced = clahe.apply(image)
    
    logger.debug(f"Applied CLAHE with clip_limit={clip_limit}")
    return enhanced


def sharpen_image(
    image: np.ndarray,
    strength: float = 1.5,
    kernel_size: int = 5
) -> np.ndarray:
    """
    Apply unsharp masking to sharpen text edges.
    
    Sharpening improves text definition, especially for slightly
    blurred or low-resolution scans.
    
    Args:
        image: Input image (BGR)
        strength: Sharpening strength (1.0 = original, 2.0 = strong)
        kernel_size: Size of Gaussian blur kernel
        
    Returns:
        Sharpened image
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV is required for image enhancement")
    
    # Create Gaussian blur
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    # Unsharp masking: original + strength * (original - blurred)
    sharpened = cv2.addWeighted(image, strength, blurred, -(strength - 1.0), 0)
    
    logger.debug(f"Applied unsharp mask with strength={strength}")
    return sharpened


def binarize_adaptive(
    image: np.ndarray,
    block_size: int = 11,
    c: int = 2
) -> np.ndarray:
    """
    Apply adaptive thresholding for binarization.
    
    Useful for documents with uneven lighting or as preprocessing
    for specific OCR engines.
    
    Args:
        image: Input image (BGR or grayscale)
        block_size: Size of pixel neighborhood for threshold calculation
        c: Constant subtracted from mean
        
    Returns:
        Binary image
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV is required for image enhancement")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Ensure block_size is odd
    if block_size % 2 == 0:
        block_size += 1
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c
    )
    
    logger.debug(f"Applied adaptive binarization with block_size={block_size}")
    return binary


def estimate_noise_level(image: np.ndarray) -> float:
    """
    Estimate the noise level in an image using Laplacian variance.
    
    Args:
        image: Input image (BGR or grayscale)
        
    Returns:
        Noise level estimate (higher = more noise)
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV is required for image enhancement")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Compute Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    
    return variance


def detect_content_type(
    image: np.ndarray,
    threshold: float = 0.6
) -> Tuple[ContentType, float]:
    """
    Detect whether content is printed, handwritten, or mixed.
    
    Uses stroke analysis to distinguish between regular printed text
    (consistent stroke width) and handwritten text (variable strokes).
    
    Args:
        image: Input image (BGR)
        threshold: Confidence threshold for classification
        
    Returns:
        Tuple of (content_type, confidence)
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV is required for content type detection")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) < 10:
        return ContentType.UNKNOWN, 0.0
    
    # Analyze stroke characteristics
    stroke_widths = []
    aspect_ratios = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 20:  # Skip noise
            continue
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        if h == 0 or w == 0:
            continue
        
        aspect_ratio = w / h
        aspect_ratios.append(aspect_ratio)
        
        # Estimate stroke width using area and perimeter
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            stroke_width = area / perimeter
            stroke_widths.append(stroke_width)
    
    if len(stroke_widths) < 5:
        return ContentType.UNKNOWN, 0.0
    
    # Calculate statistics
    stroke_std = np.std(stroke_widths)
    stroke_mean = np.mean(stroke_widths)
    aspect_std = np.std(aspect_ratios)
    
    # Coefficient of variation for stroke width
    stroke_cv = stroke_std / stroke_mean if stroke_mean > 0 else 0
    
    # Handwritten text typically has:
    # - Higher stroke width variation (CV > 0.5)
    # - Higher aspect ratio variation
    # - Less regular shapes
    
    if stroke_cv > 0.7 and aspect_std > 0.5:
        content_type = ContentType.HANDWRITTEN
        confidence = min(stroke_cv / 1.0, 1.0)
    elif stroke_cv < 0.4 and aspect_std < 0.4:
        content_type = ContentType.PRINTED
        confidence = 1.0 - stroke_cv
    else:
        content_type = ContentType.MIXED
        confidence = 0.5 + abs(0.5 - stroke_cv)
    
    # Normalize confidence
    confidence = min(max(confidence, 0.0), 1.0)
    
    logger.debug(f"Content type: {content_type.value}, confidence: {confidence:.2f}")
    return content_type, confidence


def assess_readability(image: np.ndarray) -> Tuple[str, list]:
    """
    Assess the readability/quality of a document image.
    
    Args:
        image: Input image (BGR)
        
    Returns:
        Tuple of (readability: "good"/"fair"/"poor", issues: list of strings)
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV is required for readability assessment")
    
    issues = []
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Check contrast
    contrast = gray.std()
    if contrast < 30:
        issues.append("low_contrast")
    
    # Check blur using Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 100:
        issues.append("blurry")
    
    # Check brightness
    mean_brightness = gray.mean()
    if mean_brightness < 50:
        issues.append("too_dark")
    elif mean_brightness > 230:
        issues.append("too_bright")
    
    # Check for extreme noise
    noise_level = estimate_noise_level(image)
    if noise_level > 1000:
        issues.append("high_noise")
    
    # Check image size
    h, w = image.shape[:2]
    if h < 500 or w < 500:
        issues.append("low_resolution")
    
    # Determine overall readability
    if len(issues) == 0:
        readability = "good"
    elif len(issues) <= 2:
        readability = "fair"
    else:
        readability = "poor"
    
    logger.debug(f"Readability: {readability}, issues: {issues}")
    return readability, issues


def preprocess_for_ocr(
    image: np.ndarray,
    denoise: bool = True,
    clahe: bool = True,
    sharpen: bool = True,
    deskew: bool = True,
    dewarp: bool = True
) -> np.ndarray:
    """
    Apply full preprocessing pipeline for optimal OCR accuracy.
    
    Order of operations is optimized for best results:
    1. Dewarp (perspective correction)
    2. Deskew (rotation correction)
    3. Denoise
    4. CLAHE contrast enhancement
    5. Sharpen
    
    Args:
        image: Input image (BGR)
        denoise: Apply NLM denoising
        clahe: Apply CLAHE contrast enhancement
        sharpen: Apply unsharp masking
        deskew: Apply rotation correction
        dewarp: Apply perspective correction
        
    Returns:
        Preprocessed image
    """
    from docvision.preprocess.geometry import (
        detect_page_contour,
        perspective_correction,
        deskew_image
    )
    
    result = image.copy()
    
    # 1. Dewarp if enabled and document contour detected
    if dewarp:
        contour = detect_page_contour(result)
        if contour is not None:
            result = perspective_correction(result, contour)
    
    # 2. Deskew
    if deskew:
        result = deskew_image(result)
    
    # 3. Denoise
    if denoise:
        result = denoise_image(result)
    
    # 4. CLAHE contrast enhancement
    if clahe:
        result = apply_clahe(result)
    
    # 5. Sharpen
    if sharpen:
        result = sharpen_image(result)
    
    logger.debug(f"Preprocessing complete: denoise={denoise}, clahe={clahe}, sharpen={sharpen}, deskew={deskew}, dewarp={dewarp}")
    return result
