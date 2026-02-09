"""
Image loading utilities for DocVision.

Handles various image formats with EXIF orientation correction
and format normalization for downstream processing.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
from loguru import logger

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image, ExifTags
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# Supported image formats
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}


class ImageLoader:
    """
    Load and normalize images for OCR processing.
    
    Handles EXIF orientation, format conversion, and basic validation.
    """
    
    def __init__(self, auto_orient: bool = True):
        """
        Initialize image loader.
        
        Args:
            auto_orient: Automatically correct EXIF orientation
        """
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV is required. Install with: pip install opencv-python")
        
        self.auto_orient = auto_orient
    
    def load(self, image_path: str) -> np.ndarray:
        """
        Load image from file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image as numpy array (BGR format)
        """
        path = Path(image_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        if path.suffix.lower() not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported image format: {path.suffix}")
        
        logger.debug(f"Loading image: {image_path}")
        
        # Load with PIL for EXIF handling, then convert to OpenCV
        if PIL_AVAILABLE and self.auto_orient:
            image = self._load_with_pil(image_path)
        else:
            image = cv2.imread(str(image_path))
        
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        logger.debug(f"Loaded image: {image.shape}")
        return image
    
    def _load_with_pil(self, image_path: str) -> np.ndarray:
        """
        Load image with PIL for EXIF orientation handling.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image as numpy array (BGR format)
        """
        pil_image = Image.open(image_path)
        
        # Handle EXIF orientation
        try:
            exif = pil_image._getexif()
            if exif is not None:
                orientation_key = None
                for key, value in ExifTags.TAGS.items():
                    if value == "Orientation":
                        orientation_key = key
                        break
                
                if orientation_key and orientation_key in exif:
                    orientation = exif[orientation_key]
                    pil_image = self._apply_exif_orientation(pil_image, orientation)
        except (AttributeError, KeyError, IndexError):
            pass  # No EXIF data or orientation
        
        # Convert to RGB if needed
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        
        # Convert to numpy array (RGB)
        image = np.array(pil_image)
        
        # Convert RGB to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return image
    
    def _apply_exif_orientation(self, image: "Image.Image", orientation: int) -> "Image.Image":
        """
        Apply EXIF orientation transformation.
        
        Args:
            image: PIL Image
            orientation: EXIF orientation value (1-8)
            
        Returns:
            Correctly oriented PIL Image
        """
        if orientation == 1:
            return image  # Normal
        elif orientation == 2:
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            return image.rotate(180)
        elif orientation == 4:
            return image.transpose(Image.FLIP_TOP_BOTTOM)
        elif orientation == 5:
            return image.transpose(Image.FLIP_LEFT_RIGHT).rotate(270)
        elif orientation == 6:
            return image.rotate(270, expand=True)
        elif orientation == 7:
            return image.transpose(Image.FLIP_LEFT_RIGHT).rotate(90)
        elif orientation == 8:
            return image.rotate(90, expand=True)
        return image
    
    def load_from_bytes(self, data: bytes) -> np.ndarray:
        """
        Load image from bytes.
        
        Args:
            data: Image data as bytes
            
        Returns:
            Image as numpy array (BGR format)
        """
        nparr = np.frombuffer(data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image from bytes")
        
        return image
    
    def load_from_base64(self, base64_string: str) -> np.ndarray:
        """
        Load image from base64 encoded string.
        
        Args:
            base64_string: Base64 encoded image data
            
        Returns:
            Image as numpy array (BGR format)
        """
        import base64
        
        # Remove data URL prefix if present
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        
        data = base64.b64decode(base64_string)
        return self.load_from_bytes(data)


def load_image(
    source: Union[str, bytes, np.ndarray],
    auto_orient: bool = True
) -> np.ndarray:
    """
    Convenience function to load image from various sources.
    
    Args:
        source: File path, bytes, or existing numpy array
        auto_orient: Automatically correct EXIF orientation
        
    Returns:
        Image as numpy array (BGR format)
    """
    if isinstance(source, np.ndarray):
        return source
    
    loader = ImageLoader(auto_orient=auto_orient)
    
    if isinstance(source, bytes):
        return loader.load_from_bytes(source)
    elif isinstance(source, str):
        # Check if it's a base64 string
        if source.startswith("data:image") or len(source) > 500:
            try:
                return loader.load_from_base64(source)
            except:
                pass
        # Try as file path
        return loader.load(source)
    
    raise ValueError(f"Unsupported image source type: {type(source)}")


def save_image(image: np.ndarray, output_path: str, quality: int = 95) -> None:
    """
    Save image to file.
    
    Args:
        image: Image as numpy array (BGR format)
        output_path: Output file path
        quality: JPEG quality (0-100)
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    params = []
    if path.suffix.lower() in [".jpg", ".jpeg"]:
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif path.suffix.lower() == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 9 - (quality // 11)]
    
    cv2.imwrite(str(path), image, params)
    logger.debug(f"Saved image: {output_path}")


def get_image_info(image: np.ndarray) -> dict:
    """
    Get basic image information.
    
    Args:
        image: Image as numpy array
        
    Returns:
        Dictionary with image properties
    """
    return {
        "height": image.shape[0],
        "width": image.shape[1],
        "channels": image.shape[2] if len(image.shape) > 2 else 1,
        "dtype": str(image.dtype),
        "size_bytes": image.nbytes,
    }


def resize_image(
    image: np.ndarray,
    max_size: Optional[int] = None,
    scale: Optional[float] = None
) -> Tuple[np.ndarray, float]:
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: Input image
        max_size: Maximum dimension (width or height)
        scale: Scale factor
        
    Returns:
        Tuple of (resized image, actual scale factor)
    """
    h, w = image.shape[:2]
    
    if scale is not None:
        actual_scale = scale
    elif max_size is not None:
        actual_scale = max_size / max(h, w)
        if actual_scale >= 1.0:
            return image, 1.0
    else:
        return image, 1.0
    
    new_w = int(w * actual_scale)
    new_h = int(h * actual_scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, actual_scale
