"""
Targeted re-OCR for low-confidence text regions.

This module identifies text lines with low confidence scores and re-processes
them with enhanced preprocessing and multiple OCR engines to improve accuracy.

Supports both Local and Azure modes:
- Local: Aggressive preprocessing + multi-engine ensemble (TrOCR, Tesseract)
- Azure: Optional re-crop and re-send to Document Intelligence

Usage:
    ```python
    from docvision.ocr.targeted_reocr import TargetedReOCR, ReOCRConfig
    
    reocr = TargetedReOCR(config=ReOCRConfig(confidence_threshold=0.70))
    improved_lines = reocr.process_local(image, text_lines)
    ```
"""

from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
import numpy as np
from loguru import logger
from enum import Enum

from docvision.types import TextLine, BoundingBox, ContentType, SourceEngine


class ReOCRStrategy(str, Enum):
    """Strategy for re-OCR processing."""
    ENSEMBLE = "ensemble"          # Use both TrOCR and Tesseract, pick best
    TROCR_ONLY = "trocr_only"     # Only use TrOCR with enhanced preprocessing
    TESSERACT_ONLY = "tesseract"  # Only use Tesseract with enhanced preprocessing
    SEQUENTIAL = "sequential"      # Try TrOCR first, fallback to Tesseract if still low


@dataclass
class ReOCRConfig:
    """Configuration for targeted re-OCR."""
    # Confidence thresholds
    confidence_threshold: float = 0.70         # Lines below this are re-processed
    improvement_threshold: float = 0.05        # Minimum confidence improvement to accept
    
    # Processing strategy
    strategy: ReOCRStrategy = ReOCRStrategy.ENSEMBLE
    
    # Crop settings
    padding_px: int = 8                        # Padding around text region for crop
    min_crop_size: Tuple[int, int] = (10, 10)  # Minimum (height, width) for valid crop
    
    # Enhanced preprocessing for re-OCR (more aggressive than normal)
    enhanced_denoise_strength: int = 15        # Higher than default (10)
    enhanced_clahe_clip: float = 3.0           # Higher than default (2.0)
    enhanced_sharpen_strength: float = 2.0     # Higher than default (1.5)
    apply_binarization: bool = True            # Apply adaptive binarization
    apply_morphology: bool = True              # Apply morphological operations
    
    # Scaling
    scale_factor: float = 2.0                  # Upscale crop before OCR
    
    # Azure re-OCR settings (for Azure mode)
    azure_retry_enabled: bool = True           # Enable re-send to Azure DI
    azure_retry_threshold: float = 0.50        # Lower threshold for Azure retry
    azure_max_retries: int = 1                 # Max retries per region
    
    # Limits
    max_reocr_lines: int = 50                  # Max lines to re-process per page


@dataclass
class ReOCRResult:
    """Result of re-OCR processing."""
    original_text: str
    original_confidence: float
    new_text: str
    new_confidence: float
    improved: bool
    source: SourceEngine
    engine_results: Dict[str, Tuple[str, float]] = field(default_factory=dict)


class TargetedReOCR:
    """
    Targeted re-OCR processor for low-confidence regions.
    
    Identifies text lines with confidence below threshold and re-processes
    them using enhanced preprocessing and multiple OCR engines.
    """
    
    def __init__(
        self,
        config: Optional[ReOCRConfig] = None,
        trocr_recognizer: Optional[Any] = None,
        tesseract_recognizer: Optional[Any] = None,
        device: str = "cpu"
    ):
        """
        Initialize targeted re-OCR processor.
        
        Args:
            config: Re-OCR configuration
            trocr_recognizer: Pre-initialized TrOCR recognizer (optional)
            tesseract_recognizer: Pre-initialized Tesseract recognizer (optional)
            device: Device for TrOCR inference ('cpu' or 'cuda')
        """
        self.config = config or ReOCRConfig()
        self.device = device
        
        # Use provided recognizers or load lazily
        self._trocr = trocr_recognizer
        self._tesseract = tesseract_recognizer
        
        # Stats tracking
        self.stats = {
            "total_processed": 0,
            "improved": 0,
            "failed": 0,
            "by_engine": {}
        }
        
        logger.info(
            f"TargetedReOCR initialized with threshold={self.config.confidence_threshold}, "
            f"strategy={self.config.strategy.value}"
        )
    
    @property
    def trocr(self):
        """Lazy load TrOCR recognizer."""
        if self._trocr is None:
            try:
                from docvision.ocr.trocr import TrOCRRecognizer
                self._trocr = TrOCRRecognizer(device=self.device)
            except Exception as e:
                logger.warning(f"TrOCR not available for re-OCR: {e}")
                self._trocr = False
        return self._trocr if self._trocr is not False else None
    
    @property
    def tesseract(self):
        """Lazy load Tesseract recognizer."""
        if self._tesseract is None:
            try:
                from docvision.ocr.tesseract import TesseractRecognizer
                self._tesseract = TesseractRecognizer()
            except Exception as e:
                logger.warning(f"Tesseract not available for re-OCR: {e}")
                self._tesseract = False
        return self._tesseract if self._tesseract is not False else None
    
    def identify_low_confidence_lines(
        self,
        text_lines: List[TextLine],
        threshold: Optional[float] = None
    ) -> List[TextLine]:
        """
        Identify text lines below confidence threshold.
        
        Args:
            text_lines: List of text lines to filter
            threshold: Override confidence threshold (uses config default if None)
            
        Returns:
            List of low-confidence text lines
        """
        thresh = threshold or self.config.confidence_threshold
        low_conf = [line for line in text_lines if line.confidence < thresh]
        
        # Respect max lines limit
        if len(low_conf) > self.config.max_reocr_lines:
            # Sort by confidence (lowest first) and take worst ones
            low_conf.sort(key=lambda l: l.confidence)
            low_conf = low_conf[:self.config.max_reocr_lines]
            logger.info(
                f"Limited re-OCR to {self.config.max_reocr_lines} lines "
                f"(had {len(low_conf)} below threshold)"
            )
        
        return low_conf
    
    def crop_region(
        self,
        image: np.ndarray,
        bbox: BoundingBox,
        padding: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        Crop a region from image with padding.
        
        Args:
            image: Source image (BGR)
            bbox: Bounding box to crop
            padding: Pixel padding (uses config default if None)
            
        Returns:
            Cropped image or None if invalid
        """
        pad = padding if padding is not None else self.config.padding_px
        h, w = image.shape[:2]
        
        # Calculate crop coordinates with padding
        x1 = max(0, int(bbox.x1) - pad)
        y1 = max(0, int(bbox.y1) - pad)
        x2 = min(w, int(bbox.x2) + pad)
        y2 = min(h, int(bbox.y2) + pad)
        
        # Validate size
        crop_h = y2 - y1
        crop_w = x2 - x1
        min_h, min_w = self.config.min_crop_size
        
        if crop_h < min_h or crop_w < min_w:
            logger.debug(f"Crop too small: {crop_w}x{crop_h} < {min_w}x{min_h}")
            return None
        
        return image[y1:y2, x1:x2].copy()
    
    def apply_enhanced_preprocessing(self, crop: np.ndarray) -> np.ndarray:
        """
        Apply aggressive preprocessing for better OCR on difficult regions.
        
        Args:
            crop: Input image crop (BGR)
            
        Returns:
            Enhanced image
        """
        try:
            import cv2
        except ImportError:
            logger.warning("OpenCV not available for enhanced preprocessing")
            return crop
        
        enhanced = crop.copy()
        
        # 1. Upscale for better detail
        if self.config.scale_factor > 1.0:
            scale = self.config.scale_factor
            new_h = int(enhanced.shape[0] * scale)
            new_w = int(enhanced.shape[1] * scale)
            enhanced = cv2.resize(enhanced, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # 2. Aggressive denoising
        try:
            from docvision.preprocess.enhance import denoise_image
            enhanced = denoise_image(
                enhanced,
                strength=self.config.enhanced_denoise_strength
            )
        except ImportError:
            # Fallback to OpenCV denoising
            if len(enhanced.shape) == 3:
                enhanced = cv2.fastNlMeansDenoisingColored(
                    enhanced, None,
                    self.config.enhanced_denoise_strength,
                    self.config.enhanced_denoise_strength,
                    7, 21
                )
            else:
                enhanced = cv2.fastNlMeansDenoising(
                    enhanced, None,
                    h=self.config.enhanced_denoise_strength
                )
        
        # 3. Enhanced CLAHE
        try:
            from docvision.preprocess.enhance import apply_clahe
            enhanced = apply_clahe(
                enhanced,
                clip_limit=self.config.enhanced_clahe_clip
            )
        except ImportError:
            # Fallback
            if len(enhanced.shape) == 3:
                lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(
                    clipLimit=self.config.enhanced_clahe_clip,
                    tileGridSize=(8, 8)
                )
                l = clahe.apply(l)
                lab = cv2.merge([l, a, b])
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # 4. Sharpening
        try:
            from docvision.preprocess.enhance import sharpen_image
            enhanced = sharpen_image(
                enhanced,
                strength=self.config.enhanced_sharpen_strength
            )
        except ImportError:
            # Fallback unsharp mask
            blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
            enhanced = cv2.addWeighted(
                enhanced, self.config.enhanced_sharpen_strength,
                blurred, -(self.config.enhanced_sharpen_strength - 1.0),
                0
            )
        
        # 5. Optional binarization
        if self.config.apply_binarization:
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY) if len(enhanced.shape) == 3 else enhanced
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2
            )
            # Convert back to BGR for consistency
            enhanced = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        # 6. Optional morphological cleanup
        if self.config.apply_morphology:
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY) if len(enhanced.shape) == 3 else enhanced
            # Small opening to remove noise
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cleaned = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            enhanced = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
        
        return enhanced
    
    def _recognize_with_trocr(
        self,
        crop: np.ndarray,
        content_type: ContentType = ContentType.UNKNOWN
    ) -> Tuple[str, float]:
        """Recognize using TrOCR."""
        if not self.trocr:
            return "", 0.0
        try:
            return self.trocr.recognize(crop, content_type)
        except Exception as e:
            logger.debug(f"TrOCR recognition failed: {e}")
            return "", 0.0
    
    def _recognize_with_tesseract(
        self,
        crop: np.ndarray
    ) -> Tuple[str, float]:
        """Recognize using Tesseract."""
        if not self.tesseract:
            return "", 0.0
        try:
            return self.tesseract.recognize_line(crop)
        except Exception as e:
            logger.debug(f"Tesseract recognition failed: {e}")
            return "", 0.0
    
    def reocr_line(
        self,
        image: np.ndarray,
        line: TextLine,
        content_type: ContentType = ContentType.UNKNOWN
    ) -> ReOCRResult:
        """
        Re-process a single text line with enhanced preprocessing.
        
        Args:
            image: Full page image (BGR)
            line: Text line to re-process
            content_type: Content type hint
            
        Returns:
            ReOCRResult with original and new values
        """
        result = ReOCRResult(
            original_text=line.text,
            original_confidence=line.confidence,
            new_text=line.text,
            new_confidence=line.confidence,
            improved=False,
            source=line.source
        )
        
        # Crop the region
        crop = self.crop_region(image, line.bbox)
        if crop is None:
            return result
        
        # Apply enhanced preprocessing
        enhanced_crop = self.apply_enhanced_preprocessing(crop)
        
        # Recognition based on strategy
        strategy = self.config.strategy
        
        if strategy == ReOCRStrategy.ENSEMBLE:
            # Run both engines
            trocr_text, trocr_conf = self._recognize_with_trocr(enhanced_crop, content_type)
            tess_text, tess_conf = self._recognize_with_tesseract(enhanced_crop)
            
            result.engine_results["trocr"] = (trocr_text, trocr_conf)
            result.engine_results["tesseract"] = (tess_text, tess_conf)
            
            # Pick best result
            best_text, best_conf, best_source = line.text, line.confidence, line.source
            
            if trocr_text.strip() and trocr_conf > best_conf:
                best_text, best_conf, best_source = trocr_text, trocr_conf, SourceEngine.TROCR
            if tess_text.strip() and tess_conf > best_conf:
                best_text, best_conf, best_source = tess_text, tess_conf, SourceEngine.TESSERACT
            
            result.new_text = best_text
            result.new_confidence = best_conf
            result.source = best_source
        
        elif strategy == ReOCRStrategy.TROCR_ONLY:
            text, conf = self._recognize_with_trocr(enhanced_crop, content_type)
            result.engine_results["trocr"] = (text, conf)
            if text.strip() and conf > line.confidence:
                result.new_text = text
                result.new_confidence = conf
                result.source = SourceEngine.TROCR
        
        elif strategy == ReOCRStrategy.TESSERACT_ONLY:
            text, conf = self._recognize_with_tesseract(enhanced_crop)
            result.engine_results["tesseract"] = (text, conf)
            if text.strip() and conf > line.confidence:
                result.new_text = text
                result.new_confidence = conf
                result.source = SourceEngine.TESSERACT
        
        elif strategy == ReOCRStrategy.SEQUENTIAL:
            # Try TrOCR first
            trocr_text, trocr_conf = self._recognize_with_trocr(enhanced_crop, content_type)
            result.engine_results["trocr"] = (trocr_text, trocr_conf)
            
            if trocr_text.strip() and trocr_conf > line.confidence + self.config.improvement_threshold:
                result.new_text = trocr_text
                result.new_confidence = trocr_conf
                result.source = SourceEngine.TROCR
            else:
                # Fallback to Tesseract
                tess_text, tess_conf = self._recognize_with_tesseract(enhanced_crop)
                result.engine_results["tesseract"] = (tess_text, tess_conf)
                
                if tess_text.strip() and tess_conf > line.confidence:
                    result.new_text = tess_text
                    result.new_confidence = tess_conf
                    result.source = SourceEngine.TESSERACT
        
        # Check if improved
        improvement = result.new_confidence - result.original_confidence
        result.improved = improvement >= self.config.improvement_threshold
        
        return result
    
    def process_local(
        self,
        image: np.ndarray,
        text_lines: List[TextLine],
        content_type: ContentType = ContentType.UNKNOWN,
        in_place: bool = True
    ) -> List[TextLine]:
        """
        Process low-confidence lines using local OCR engines.
        
        Args:
            image: Full page image (BGR)
            text_lines: All text lines from OCR
            content_type: Content type hint
            in_place: If True, modify lines in place; if False, return copies
            
        Returns:
            Updated text lines (same list if in_place=True)
        """
        # Identify candidates
        low_conf_lines = self.identify_low_confidence_lines(text_lines)
        
        if not low_conf_lines:
            logger.debug("No low-confidence lines found for re-OCR")
            return text_lines
        
        logger.info(
            f"Processing {len(low_conf_lines)} low-confidence lines "
            f"(threshold: {self.config.confidence_threshold})"
        )
        
        # Track results
        improved_count = 0
        results: List[ReOCRResult] = []
        
        for line in low_conf_lines:
            result = self.reocr_line(image, line, content_type)
            results.append(result)
            
            if result.improved:
                logger.debug(
                    f"Improved: '{result.original_text}' ({result.original_confidence:.2f}) -> "
                    f"'{result.new_text}' ({result.new_confidence:.2f}) via {result.source.value}"
                )
                
                # Update line
                line.text = result.new_text
                line.confidence = result.new_confidence
                line.source = result.source
                improved_count += 1
                
                # Track by engine
                engine = result.source.value
                self.stats["by_engine"][engine] = self.stats["by_engine"].get(engine, 0) + 1
        
        # Update stats
        self.stats["total_processed"] += len(low_conf_lines)
        self.stats["improved"] += improved_count
        self.stats["failed"] += len(low_conf_lines) - improved_count
        
        logger.info(
            f"Re-OCR complete: {improved_count}/{len(low_conf_lines)} lines improved "
            f"({improved_count/len(low_conf_lines)*100:.1f}%)"
        )
        
        return text_lines
    
    def process_azure(
        self,
        image: np.ndarray,
        text_lines: List[TextLine],
        azure_client: Optional[Any] = None,
        in_place: bool = True
    ) -> List[TextLine]:
        """
        Process low-confidence lines by re-sending crops to Azure DI.
        
        This is optional and adds API cost. Use sparingly for critical documents.
        
        Args:
            image: Full page image (BGR)
            text_lines: All text lines from Azure DI
            azure_client: Azure Document Intelligence client
            in_place: If True, modify lines in place
            
        Returns:
            Updated text lines
        """
        if not self.config.azure_retry_enabled:
            logger.debug("Azure re-OCR disabled in config")
            return text_lines
        
        if azure_client is None:
            logger.warning("Azure client not provided, skipping Azure re-OCR")
            return text_lines
        
        # Use lower threshold for Azure retry (more selective)
        low_conf_lines = self.identify_low_confidence_lines(
            text_lines,
            threshold=self.config.azure_retry_threshold
        )
        
        if not low_conf_lines:
            logger.debug("No lines below Azure retry threshold")
            return text_lines
        
        logger.info(
            f"Re-sending {len(low_conf_lines)} regions to Azure DI "
            f"(threshold: {self.config.azure_retry_threshold})"
        )
        
        improved_count = 0
        
        for line in low_conf_lines:
            # Crop with extra padding for Azure
            crop = self.crop_region(image, line.bbox, padding=12)
            if crop is None:
                continue
            
            try:
                # Convert to bytes for Azure
                import cv2
                _, buffer = cv2.imencode('.png', crop)
                image_bytes = buffer.tobytes()
                
                # Call Azure DI
                poller = azure_client.begin_analyze_document(
                    "prebuilt-read",
                    image_bytes
                )
                result = poller.result()
                
                # Extract text from result
                if result.content and result.content.strip():
                    new_text = result.content.strip()
                    # Azure doesn't give per-line confidence, estimate based on result
                    new_conf = 0.85  # Assume high confidence from Azure re-read
                    
                    if new_conf > line.confidence:
                        logger.debug(
                            f"Azure improved: '{line.text}' -> '{new_text}'"
                        )
                        line.text = new_text
                        line.confidence = new_conf
                        line.source = SourceEngine.AZURE_DOC_INTELLIGENCE
                        improved_count += 1
                        
            except Exception as e:
                logger.debug(f"Azure re-OCR failed for region: {e}")
        
        if improved_count > 0:
            logger.info(f"Azure re-OCR improved {improved_count} lines")
        
        return text_lines
    
    def process(
        self,
        image: np.ndarray,
        text_lines: List[TextLine],
        mode: str = "local",
        content_type: ContentType = ContentType.UNKNOWN,
        azure_client: Optional[Any] = None,
        in_place: bool = True
    ) -> List[TextLine]:
        """
        Main entry point for targeted re-OCR.
        
        Args:
            image: Full page image (BGR)
            text_lines: All text lines from initial OCR
            mode: Processing mode ("local", "azure", or "hybrid")
            content_type: Content type hint
            azure_client: Azure DI client (required for azure/hybrid modes)
            in_place: If True, modify lines in place
            
        Returns:
            Updated text lines
        """
        if mode == "local":
            return self.process_local(image, text_lines, content_type, in_place)
        elif mode == "azure":
            return self.process_azure(image, text_lines, azure_client, in_place)
        elif mode == "hybrid":
            # First try local engines, then Azure for remaining low-conf
            text_lines = self.process_local(image, text_lines, content_type, in_place)
            return self.process_azure(image, text_lines, azure_client, in_place)
        else:
            logger.warning(f"Unknown re-OCR mode: {mode}, using local")
            return self.process_local(image, text_lines, content_type, in_place)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = self.stats.copy()
        if stats["total_processed"] > 0:
            stats["improvement_rate"] = stats["improved"] / stats["total_processed"]
        else:
            stats["improvement_rate"] = 0.0
        return stats
    
    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self.stats = {
            "total_processed": 0,
            "improved": 0,
            "failed": 0,
            "by_engine": {}
        }
