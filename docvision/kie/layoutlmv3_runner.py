"""
LayoutLMv3 for token-based Key Information Extraction.

Uses multimodal Transformer (text + image + layout) for
sequence labeling (NER) and relation extraction.
"""

from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import numpy as np
from loguru import logger

from docvision.types import (
    Field, Candidate, BoundingBox, SourceEngine, FieldStatus,
    TextLine, Word
)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import (
        LayoutLMv3Processor,
        LayoutLMv3ForTokenClassification,
        LayoutLMv3ForSequenceClassification
    )
    from PIL import Image
    LAYOUTLM_AVAILABLE = True
except ImportError:
    LAYOUTLM_AVAILABLE = False
    logger.warning("LayoutLMv3 not available. Install transformers.")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class LayoutLMv3Runner:
    """
    Run LayoutLMv3 for token-based KIE.
    
    Performs sequence labeling to classify OCR tokens into
    entity types (e.g., invoice_number, date, amount).
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/layoutlmv3-base",
        device: str = "cpu",
        labels: Optional[List[str]] = None
    ):
        """
        Initialize LayoutLMv3 runner.
        
        Args:
            model_name: HuggingFace model name or local path
            device: Device for inference ('cpu' or 'cuda')
            labels: List of entity labels (if fine-tuned model)
        """
        self.model_name = model_name
        self.device = device
        
        # Default labels for common document fields
        self.labels = labels or [
            "O",  # Outside any entity
            "B-HEADER",
            "I-HEADER",
            "B-QUESTION",
            "I-QUESTION",
            "B-ANSWER",
            "I-ANSWER",
            "B-DATE",
            "I-DATE",
            "B-AMOUNT",
            "I-AMOUNT",
            "B-INVOICE_NUMBER",
            "I-INVOICE_NUMBER",
            "B-VENDOR",
            "I-VENDOR",
            "B-ADDRESS",
            "I-ADDRESS",
            "B-TOTAL",
            "I-TOTAL",
            "B-TAX",
            "I-TAX",
            "B-ITEM",
            "I-ITEM",
        ]
        
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.id2label = {i: label for i, label in enumerate(self.labels)}
        
        self.processor = None
        self.model = None
        self.model_loaded = False
        
        if LAYOUTLM_AVAILABLE and TORCH_AVAILABLE:
            self._load_model()
    
    def _load_model(self) -> None:
        """Load LayoutLMv3 model."""
        try:
            logger.info(f"Loading LayoutLMv3 model: {self.model_name}")
            self.processor = LayoutLMv3Processor.from_pretrained(
                self.model_name,
                apply_ocr=False  # We provide our own OCR
            )
            
            # Try to load fine-tuned model, fall back to base
            try:
                self.model = LayoutLMv3ForTokenClassification.from_pretrained(
                    self.model_name,
                    num_labels=len(self.labels),
                    id2label=self.id2label,
                    label2id=self.label2id
                )
            except Exception:
                # Use base model for feature extraction
                logger.info("Loading base LayoutLMv3 (not fine-tuned for NER)")
                self.model = LayoutLMv3ForTokenClassification.from_pretrained(
                    "microsoft/layoutlmv3-base",
                    num_labels=len(self.labels),
                    id2label=self.id2label,
                    label2id=self.label2id,
                    ignore_mismatched_sizes=True
                )
            
            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            logger.info("LayoutLMv3 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load LayoutLMv3 model: {e}")
    
    def extract(
        self,
        image: np.ndarray,
        words: List[str],
        boxes: List[List[int]],
    ) -> List[Tuple[str, str, float]]:
        """
        Extract entities from OCR tokens.
        
        Args:
            image: Document image (BGR format)
            words: List of OCR words
            boxes: List of bounding boxes [x1, y1, x2, y2] normalized to 1000
            
        Returns:
            List of (word, label, confidence) tuples
        """
        if not self.model_loaded:
            logger.warning("LayoutLMv3 not loaded")
            return [(w, "O", 0.0) for w in words]
        
        if not words or not boxes:
            return []
        
        # Convert BGR to RGB
        if CV2_AVAILABLE and len(image.shape) == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        pil_image = Image.fromarray(rgb_image)
        
        # Normalize boxes to 0-1000 range
        normalized_boxes = self._normalize_boxes(boxes, image.shape)
        
        # Process inputs
        encoding = self.processor(
            pil_image,
            words,
            boxes=normalized_boxes,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"
        )
        
        # Extract word_ids before converting to plain dict (which loses
        # the BatchEncoding.word_ids() method).
        word_ids = encoding.word_ids(0)
        
        # Move to device
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**encoding)
        
        # Get predictions
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        probabilities = torch.softmax(logits, dim=-1)
        
        # Process results
        results = []
        pred_array = predictions[0].cpu().numpy()
        prob_array = probabilities[0].cpu().numpy()
        
        # Map back to original words (handling tokenization)
        current_word_idx = -1
        for i, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            
            if word_idx != current_word_idx:
                current_word_idx = word_idx
                if word_idx < len(words):
                    pred_id = pred_array[i]
                    label = self.id2label.get(pred_id, "O")
                    confidence = float(prob_array[i][pred_id])
                    results.append((words[word_idx], label, confidence))
        
        return results
    
    def _normalize_boxes(
        self,
        boxes: List[List[int]],
        image_shape: Tuple[int, ...]
    ) -> List[List[int]]:
        """Normalize bounding boxes to 0-1000 range."""
        h, w = image_shape[:2]
        
        normalized = []
        for box in boxes:
            x1, y1, x2, y2 = box
            normalized.append([
                int(x1 * 1000 / w),
                int(y1 * 1000 / h),
                int(x2 * 1000 / w),
                int(y2 * 1000 / h)
            ])
        
        return normalized
    
    def extract_from_text_lines(
        self,
        image: np.ndarray,
        text_lines: List[TextLine],
        page_num: int = 1
    ) -> List[Field]:
        """
        Extract fields from text lines.
        
        Args:
            image: Document image
            text_lines: OCR text lines with bounding boxes
            page_num: Page number for field metadata
            
        Returns:
            List of extracted fields
        """
        # Build words and boxes from text lines
        words = []
        boxes = []
        word_to_line = []  # Map word index to line index
        
        for line_idx, line in enumerate(text_lines):
            if line.words:
                # Use word-level data if available
                for word in line.words:
                    words.append(word.text)
                    boxes.append([
                        int(word.bbox.x1),
                        int(word.bbox.y1),
                        int(word.bbox.x2),
                        int(word.bbox.y2)
                    ])
                    word_to_line.append(line_idx)
            else:
                # Use line-level data
                for word_text in line.text.split():
                    words.append(word_text)
                    boxes.append([
                        int(line.bbox.x1),
                        int(line.bbox.y1),
                        int(line.bbox.x2),
                        int(line.bbox.y2)
                    ])
                    word_to_line.append(line_idx)
        
        if not words:
            return []
        
        # Run extraction
        results = self.extract(image, words, boxes)
        
        # Group consecutive B/I tags into fields
        fields = self._group_entities(results, boxes, word_to_line, text_lines, page_num)
        
        return fields
    
    def _group_entities(
        self,
        results: List[Tuple[str, str, float]],
        boxes: List[List[int]],
        word_to_line: List[int],
        text_lines: List[TextLine],
        page_num: int
    ) -> List[Field]:
        """Group BIO-tagged tokens into fields."""
        fields = []
        current_entity = None
        current_words = []
        current_boxes = []
        current_confidences = []
        
        for i, (word, label, conf) in enumerate(results):
            if label.startswith("B-"):
                # Start of new entity
                if current_entity:
                    # Save previous entity
                    fields.append(self._create_field_from_entity(
                        current_entity,
                        current_words,
                        current_boxes,
                        current_confidences,
                        page_num
                    ))
                
                current_entity = label[2:]  # Remove B- prefix
                current_words = [word]
                current_boxes = [boxes[i]] if i < len(boxes) else []
                current_confidences = [conf]
            
            elif label.startswith("I-") and current_entity:
                entity_type = label[2:]
                if entity_type == current_entity:
                    # Continue current entity
                    current_words.append(word)
                    if i < len(boxes):
                        current_boxes.append(boxes[i])
                    current_confidences.append(conf)
            
            else:
                # Outside or different entity
                if current_entity:
                    fields.append(self._create_field_from_entity(
                        current_entity,
                        current_words,
                        current_boxes,
                        current_confidences,
                        page_num
                    ))
                    current_entity = None
                    current_words = []
                    current_boxes = []
                    current_confidences = []
        
        # Don't forget last entity
        if current_entity:
            fields.append(self._create_field_from_entity(
                current_entity,
                current_words,
                current_boxes,
                current_confidences,
                page_num
            ))
        
        return fields
    
    def _create_field_from_entity(
        self,
        entity_type: str,
        words: List[str],
        boxes: List[List[int]],
        confidences: List[float],
        page_num: int
    ) -> Field:
        """Create a Field object from grouped entity tokens."""
        value = " ".join(words)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Calculate combined bounding box
        bbox = None
        if boxes:
            x1 = min(b[0] for b in boxes)
            y1 = min(b[1] for b in boxes)
            x2 = max(b[2] for b in boxes)
            y2 = max(b[3] for b in boxes)
            bbox = BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))
        
        # Determine data type from entity type
        data_type = "string"
        if entity_type in ["AMOUNT", "TOTAL", "TAX"]:
            data_type = "currency"
        elif entity_type == "DATE":
            data_type = "date"
        
        # Create candidate
        candidate = Candidate(
            source=SourceEngine.LAYOUTLMV3,
            value=value,
            confidence=avg_confidence,
            bbox=bbox,
            page=page_num
        )
        
        # Determine status
        if avg_confidence >= 0.8:
            status = FieldStatus.CONFIDENT
        elif avg_confidence >= 0.5:
            status = FieldStatus.SINGLE_SOURCE
        else:
            status = FieldStatus.UNCERTAIN
        
        return Field(
            name=entity_type.lower(),
            value=value,
            data_type=data_type,
            confidence=avg_confidence,
            status=status,
            page=page_num,
            bbox=bbox,
            chosen_source=SourceEngine.LAYOUTLMV3,
            candidates=[candidate]
        )
    
    def classify_document(
        self,
        image: np.ndarray,
        words: List[str],
        boxes: List[List[int]]
    ) -> Tuple[str, float]:
        """
        Classify document type (if model supports it).
        
        Args:
            image: Document image
            words: OCR words
            boxes: Bounding boxes
            
        Returns:
            Tuple of (document_type, confidence)
        """
        # This would require a classification head model
        # Placeholder for future implementation
        return "unknown", 0.0
