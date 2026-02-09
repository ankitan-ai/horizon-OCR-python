"""
TrOCR-based text recognition.

Uses Microsoft's TrOCR for high-quality line-level text recognition.
Supports both printed and handwritten text models.
"""

from typing import List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
from loguru import logger

from docvision.types import TextLine, BoundingBox, ContentType, SourceEngine

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from PIL import Image
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False
    logger.warning("TrOCR not available. Install transformers and PIL.")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class TrOCRRecognizer:
    """
    Text recognition using Microsoft TrOCR.
    
    Supports separate models for printed and handwritten text,
    with automatic content type detection and model routing.
    """
    
    def __init__(
        self,
        printed_model: str = "models/trocr-base-printed",
        handwritten_model: str = "models/trocr-base-handwritten",
        device: str = "cpu",
        max_length: int = 256,
        batch_size: int = 8
    ):
        """
        Initialize TrOCR recognizer.
        
        Args:
            printed_model: HuggingFace model name for printed text
            handwritten_model: HuggingFace model name for handwritten text
            device: Device for inference ('cpu' or 'cuda')
            max_length: Maximum output sequence length
            batch_size: Batch size for inference
        """
        self.printed_model_name = printed_model
        self.handwritten_model_name = handwritten_model
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        
        self.printed_processor = None
        self.printed_model = None
        self.handwritten_processor = None
        self.handwritten_model = None
        
        self.printed_loaded = False
        self.handwritten_loaded = False
        
        if TROCR_AVAILABLE and TORCH_AVAILABLE:
            self._load_models()
    
    def _load_models(self) -> None:
        """Load TrOCR models."""
        try:
            # Load printed model
            logger.info(f"Loading TrOCR printed model: {self.printed_model_name}")
            self.printed_processor = TrOCRProcessor.from_pretrained(self.printed_model_name)
            self.printed_model = VisionEncoderDecoderModel.from_pretrained(self.printed_model_name)
            self.printed_model.to(self.device)
            self.printed_model.eval()
            self.printed_loaded = True
            
            # Load handwritten model
            logger.info(f"Loading TrOCR handwritten model: {self.handwritten_model_name}")
            self.handwritten_processor = TrOCRProcessor.from_pretrained(self.handwritten_model_name)
            self.handwritten_model = VisionEncoderDecoderModel.from_pretrained(self.handwritten_model_name)
            self.handwritten_model.to(self.device)
            self.handwritten_model.eval()
            self.handwritten_loaded = True
            
            logger.info("TrOCR models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load TrOCR models: {e}")
    
    def recognize(
        self,
        image: np.ndarray,
        content_type: ContentType = ContentType.UNKNOWN
    ) -> Tuple[str, float]:
        """
        Recognize text in a single image crop.
        
        Args:
            image: Text line image (BGR format)
            content_type: Content type (printed/handwritten/unknown)
            
        Returns:
            Tuple of (recognized_text, confidence)
        """
        if not self.printed_loaded:
            logger.warning("TrOCR not loaded, returning empty result")
            return "", 0.0
        
        # Select model based on content type
        if content_type == ContentType.HANDWRITTEN and self.handwritten_loaded:
            processor = self.handwritten_processor
            model = self.handwritten_model
        else:
            processor = self.printed_processor
            model = self.printed_model
        
        # Convert to PIL Image
        if CV2_AVAILABLE and len(image.shape) == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        pil_image = Image.fromarray(rgb_image)
        
        # Process image
        pixel_values = processor(pil_image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        
        # Generate text
        with torch.inference_mode():
            generated_ids = model.generate(
                pixel_values,
                max_length=self.max_length,
                output_scores=True,
                return_dict_in_generate=True
            )
        
        # Decode text
        text = processor.batch_decode(
            generated_ids.sequences, 
            skip_special_tokens=True
        )[0]
        
        # Calculate confidence from scores
        if hasattr(generated_ids, 'scores') and generated_ids.scores:
            # Average softmax probability across tokens
            scores = torch.stack(generated_ids.scores, dim=1)
            probs = torch.softmax(scores, dim=-1)
            max_probs = probs.max(dim=-1).values
            confidence = float(max_probs.mean().cpu())
        else:
            confidence = 0.8  # Default confidence if scores unavailable
        
        return text.strip(), confidence
    
    def recognize_batch(
        self,
        images: List[np.ndarray],
        content_types: Optional[List[ContentType]] = None
    ) -> List[Tuple[str, float]]:
        """
        Recognize text in a batch of images.
        
        Args:
            images: List of text line images
            content_types: Optional list of content types per image
            
        Returns:
            List of (text, confidence) tuples
        """
        if not images:
            return []
        
        if not self.printed_loaded:
            return [("", 0.0) for _ in images]
        
        if content_types is None:
            content_types = [ContentType.UNKNOWN] * len(images)
        
        # Group by content type for efficient batching
        printed_indices = []
        handwritten_indices = []
        
        for i, ct in enumerate(content_types):
            if ct == ContentType.HANDWRITTEN and self.handwritten_loaded:
                handwritten_indices.append(i)
            else:
                printed_indices.append(i)
        
        results = [("", 0.0)] * len(images)
        
        # Process printed images
        if printed_indices:
            printed_images = [images[i] for i in printed_indices]
            printed_results = self._batch_inference(
                printed_images,
                self.printed_processor,
                self.printed_model
            )
            for idx, result in zip(printed_indices, printed_results):
                results[idx] = result
        
        # Process handwritten images
        if handwritten_indices:
            hw_images = [images[i] for i in handwritten_indices]
            hw_results = self._batch_inference(
                hw_images,
                self.handwritten_processor,
                self.handwritten_model
            )
            for idx, result in zip(handwritten_indices, hw_results):
                results[idx] = result
        
        return results
    
    def _batch_inference(
        self,
        images: List[np.ndarray],
        processor: "TrOCRProcessor",
        model: "VisionEncoderDecoderModel"
    ) -> List[Tuple[str, float]]:
        """Run batch inference with a specific model."""
        results = []
        
        # Process in batches
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]
            
            # Convert to PIL
            pil_images = []
            for img in batch_images:
                if CV2_AVAILABLE and len(img.shape) == 3:
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    rgb = img
                pil_images.append(Image.fromarray(rgb))
            
            # Process batch
            pixel_values = processor(
                pil_images,
                return_tensors="pt",
                padding=True
            ).pixel_values
            pixel_values = pixel_values.to(self.device)
            
            # Generate with scores for proper confidence
            with torch.inference_mode():
                generated_ids = model.generate(
                    pixel_values,
                    max_length=self.max_length,
                    output_scores=True,
                    return_dict_in_generate=True
                )
            
            # Decode
            texts = processor.batch_decode(
                generated_ids.sequences,
                skip_special_tokens=True
            )
            
            # Calculate per-item confidence from scores
            if hasattr(generated_ids, 'scores') and generated_ids.scores:
                scores = torch.stack(generated_ids.scores, dim=1)
                probs = torch.softmax(scores, dim=-1)
                max_probs = probs.max(dim=-1).values  # (batch, seq_len)
                
                for j, text in enumerate(texts):
                    if j < max_probs.shape[0]:
                        # Average confidence over generated tokens
                        token_probs = max_probs[j]
                        confidence = float(token_probs.mean().cpu())
                    else:
                        confidence = 0.8
                    results.append((text.strip(), confidence))
            else:
                for text in texts:
                    results.append((text.strip(), 0.8))
        
        return results
    
    def recognize_with_dual_models(
        self,
        image: np.ndarray
    ) -> Tuple[str, float, ContentType]:
        """
        Recognize using both models and return best result.
        
        Useful when content type is uncertain - runs both models
        and returns the result with higher confidence.
        
        Args:
            image: Text line image
            
        Returns:
            Tuple of (text, confidence, detected_content_type)
        """
        if not self.printed_loaded:
            return "", 0.0, ContentType.UNKNOWN
        
        # Run printed model
        printed_text, printed_conf = self.recognize(image, ContentType.PRINTED)
        
        # Run handwritten model if available
        if self.handwritten_loaded:
            hw_text, hw_conf = self.recognize(image, ContentType.HANDWRITTEN)
            
            # Choose best result
            if hw_conf > printed_conf:
                return hw_text, hw_conf, ContentType.HANDWRITTEN
        
        return printed_text, printed_conf, ContentType.PRINTED
    
    def update_text_lines(
        self,
        image: np.ndarray,
        text_lines: List[TextLine],
        content_type: ContentType = ContentType.UNKNOWN
    ) -> List[TextLine]:
        """
        Update text lines with recognized text.
        
        Args:
            image: Full document image
            text_lines: Text lines with bounding boxes
            content_type: Overall content type
            
        Returns:
            Updated text lines with recognized text
        """
        from docvision.ocr.crops import crop_text_region
        
        # Crop all regions
        crops = []
        valid_indices = []
        
        for i, line in enumerate(text_lines):
            try:
                crop = crop_text_region(image, line.bbox, padding=2)
                if crop.size > 0 and crop.shape[0] > 5 and crop.shape[1] > 5:
                    crops.append(crop)
                    valid_indices.append(i)
            except Exception as e:
                logger.debug(f"Failed to crop line {i}: {e}")
        
        # Recognize batch
        content_types = [content_type] * len(crops)
        results = self.recognize_batch(crops, content_types)
        
        # Update text lines
        for idx, (text, conf) in zip(valid_indices, results):
            text_lines[idx].text = text
            text_lines[idx].confidence = conf
            text_lines[idx].source = SourceEngine.TROCR
        
        return text_lines
