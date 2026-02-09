"""
Tesseract OCR integration.

Provides Tesseract as a backup OCR engine with word-level confidences.
Used when TrOCR confidence is low or for comparison in ensemble.
"""

from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import numpy as np
from loguru import logger

from docvision.types import TextLine, Word, BoundingBox, SourceEngine

try:
    import pytesseract
    from pytesseract import Output
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("pytesseract not available. Tesseract OCR will be disabled.")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class TesseractRecognizer:
    """
    Text recognition using Tesseract OCR.
    
    Provides word-level recognition with confidence scores.
    Used as backup engine and for ensemble voting.
    """
    
    def __init__(
        self,
        lang: str = "eng",
        config: str = "--oem 3 --psm 6",
        min_confidence: float = 0.0
    ):
        """
        Initialize Tesseract recognizer.
        
        Args:
            lang: Tesseract language code (e.g., 'eng', 'eng+fra')
            config: Tesseract configuration string
            min_confidence: Minimum word confidence to include
        """
        if not TESSERACT_AVAILABLE:
            raise ImportError("pytesseract is required. Install with: pip install pytesseract")
        
        self.lang = lang
        self.config = config
        self.min_confidence = min_confidence
        
        # Auto-detect Tesseract binary on Windows if not in PATH
        import platform
        if platform.system() == "Windows":
            import shutil
            if not shutil.which("tesseract"):
                common_paths = [
                    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                ]
                for path in common_paths:
                    if Path(path).exists():
                        pytesseract.pytesseract.tesseract_cmd = path
                        logger.info(f"Auto-detected Tesseract at: {path}")
                        break
        
        # Verify Tesseract installation
        try:
            pytesseract.get_tesseract_version()
            logger.info(f"Tesseract initialized with lang={lang}")
        except Exception as e:
            logger.error(f"Tesseract not properly installed: {e}")
            raise
    
    def recognize(
        self,
        image: np.ndarray,
        psm: Optional[int] = None
    ) -> Tuple[str, float]:
        """
        Recognize text in an image.
        
        Args:
            image: Input image (BGR or grayscale)
            psm: Page segmentation mode (overrides default)
            
        Returns:
            Tuple of (recognized_text, average_confidence)
        """
        # Prepare image
        processed = self._preprocess_image(image)
        
        # Build config
        config = self.config
        if psm is not None:
            config = f"--oem 3 --psm {psm}"
        
        # Get detailed OCR data
        data = pytesseract.image_to_data(
            processed,
            lang=self.lang,
            config=config,
            output_type=Output.DICT
        )
        
        # Extract text and confidence
        words = []
        confidences = []
        
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            text = data['text'][i].strip()
            conf = float(data['conf'][i])
            
            if text and conf >= self.min_confidence * 100:  # Tesseract uses 0-100
                words.append(text)
                confidences.append(conf / 100.0)
        
        full_text = " ".join(words)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return full_text, avg_confidence
    
    def recognize_line(
        self,
        image: np.ndarray
    ) -> Tuple[str, float]:
        """
        Recognize a single text line.
        
        Uses PSM 7 (single line) for optimal line recognition.
        
        Args:
            image: Text line image
            
        Returns:
            Tuple of (text, confidence)
        """
        return self.recognize(image, psm=7)
    
    def recognize_with_words(
        self,
        image: np.ndarray,
        offset: Tuple[int, int] = (0, 0)
    ) -> Tuple[str, float, List[Word]]:
        """
        Recognize text with word-level details.
        
        Args:
            image: Input image
            offset: (x, y) offset to add to word bounding boxes
            
        Returns:
            Tuple of (full_text, avg_confidence, list_of_words)
        """
        processed = self._preprocess_image(image)
        
        data = pytesseract.image_to_data(
            processed,
            lang=self.lang,
            config=self.config,
            output_type=Output.DICT
        )
        
        words = []
        all_text = []
        confidences = []
        
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            text = data['text'][i].strip()
            conf = float(data['conf'][i])
            
            if not text or conf < self.min_confidence * 100:
                continue
            
            x = data['left'][i] + offset[0]
            y = data['top'][i] + offset[1]
            w = data['width'][i]
            h = data['height'][i]
            
            word = Word(
                text=text,
                bbox=BoundingBox(
                    x1=float(x),
                    y1=float(y),
                    x2=float(x + w),
                    y2=float(y + h)
                ),
                confidence=conf / 100.0,
                source=SourceEngine.TESSERACT
            )
            words.append(word)
            all_text.append(text)
            confidences.append(conf / 100.0)
        
        full_text = " ".join(all_text)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return full_text, avg_confidence, words
    
    def recognize_full_page(
        self,
        image: np.ndarray
    ) -> Tuple[List[TextLine], str]:
        """
        Recognize full page with line detection.
        
        Args:
            image: Full page image
            
        Returns:
            Tuple of (text_lines, raw_text)
        """
        processed = self._preprocess_image(image)
        
        data = pytesseract.image_to_data(
            processed,
            lang=self.lang,
            config=self.config,
            output_type=Output.DICT
        )
        
        # Group words by line number
        lines_data: Dict[Tuple[int, int], List[Dict]] = {}
        
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            text = data['text'][i].strip()
            conf = float(data['conf'][i])
            
            if not text or conf < 0:
                continue
            
            block_num = data['block_num'][i]
            line_num = data['line_num'][i]
            key = (block_num, line_num)
            
            if key not in lines_data:
                lines_data[key] = []
            
            lines_data[key].append({
                'text': text,
                'conf': conf / 100.0,
                'left': data['left'][i],
                'top': data['top'][i],
                'width': data['width'][i],
                'height': data['height'][i]
            })
        
        # Build text lines
        text_lines = []
        
        for key in sorted(lines_data.keys()):
            words_data = lines_data[key]
            
            if not words_data:
                continue
            
            # Calculate line bounding box
            x1 = min(w['left'] for w in words_data)
            y1 = min(w['top'] for w in words_data)
            x2 = max(w['left'] + w['width'] for w in words_data)
            y2 = max(w['top'] + w['height'] for w in words_data)
            
            # Build words
            words = []
            for wd in words_data:
                word = Word(
                    text=wd['text'],
                    bbox=BoundingBox(
                        x1=float(wd['left']),
                        y1=float(wd['top']),
                        x2=float(wd['left'] + wd['width']),
                        y2=float(wd['top'] + wd['height'])
                    ),
                    confidence=wd['conf'],
                    source=SourceEngine.TESSERACT
                )
                words.append(word)
            
            line_text = " ".join(w['text'] for w in words_data)
            avg_conf = sum(w['conf'] for w in words_data) / len(words_data)
            
            line = TextLine(
                text=line_text,
                words=words,
                bbox=BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
                confidence=avg_conf,
                source=SourceEngine.TESSERACT
            )
            text_lines.append(line)
        
        # Build raw text
        raw_text = "\n".join(line.text for line in text_lines)
        
        return text_lines, raw_text
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better Tesseract recognition.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        if not CV2_AVAILABLE:
            return image
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply adaptive thresholding for better binarization
        # This helps with varying illumination
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        return binary
    
    def get_osd(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Get orientation and script detection.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with orientation, script, and confidence
        """
        try:
            osd = pytesseract.image_to_osd(image, output_type=Output.DICT)
            return {
                'orientation': osd.get('orientation', 0),
                'rotate': osd.get('rotate', 0),
                'script': osd.get('script', 'Latin'),
                'confidence': osd.get('orientation_conf', 0) / 100.0
            }
        except Exception as e:
            logger.warning(f"OSD failed: {e}")
            return {
                'orientation': 0,
                'rotate': 0,
                'script': 'unknown',
                'confidence': 0.0
            }
    
    @staticmethod
    def set_tesseract_path(path: str) -> None:
        """
        Set custom Tesseract executable path.
        
        Args:
            path: Path to tesseract executable
        """
        pytesseract.pytesseract.tesseract_cmd = path
    
    @staticmethod
    def get_available_languages() -> List[str]:
        """Get list of available Tesseract languages."""
        try:
            return pytesseract.get_languages()
        except Exception:
            return ["eng"]
