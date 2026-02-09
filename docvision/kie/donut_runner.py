"""
Donut (OCR-free) document understanding model.

Donut directly predicts structured JSON from document images
without requiring separate OCR. Strong for end-to-end parsing
of forms, receipts, and invoices.
"""

from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import json
import re
import numpy as np
from loguru import logger

from docvision.types import Field, Candidate, BoundingBox, SourceEngine, FieldStatus

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import DonutProcessor, VisionEncoderDecoderModel
    from PIL import Image
    DONUT_AVAILABLE = True
except ImportError:
    DONUT_AVAILABLE = False
    logger.warning("Donut not available. Install transformers.")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class DonutRunner:
    """
    Run Donut model for OCR-free document understanding.
    
    Donut directly outputs structured JSON from document images,
    making it ideal for forms and receipts with known schemas.
    """
    
    def __init__(
        self,
        model_name: str = "naver-clova-ix/donut-base-finetuned-cord-v2",
        device: str = "cpu",
        max_length: int = 512
    ):
        """
        Initialize Donut runner.
        
        Args:
            model_name: HuggingFace model name or local path
            device: Device for inference ('cpu' or 'cuda')
            max_length: Maximum output sequence length
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        
        self.processor = None
        self.model = None
        self.model_loaded = False
        
        if DONUT_AVAILABLE and TORCH_AVAILABLE:
            self._load_model()
    
    def _load_model(self) -> None:
        """Load Donut model."""
        try:
            logger.info(f"Loading Donut model: {self.model_name}")
            self.processor = DonutProcessor.from_pretrained(self.model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            logger.info("Donut model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Donut model: {e}")
    
    def extract(
        self,
        image: np.ndarray,
        task_prompt: str = "<s_cord-v2>"
    ) -> Tuple[Dict[str, Any], float]:
        """
        Extract structured information from document image.
        
        Args:
            image: Document image (BGR format)
            task_prompt: Task-specific prompt for the model
            
        Returns:
            Tuple of (extracted_data dict, overall confidence)
        """
        if not self.model_loaded:
            logger.warning("Donut not loaded, returning empty result")
            return {}, 0.0
        
        # Convert BGR to RGB
        if CV2_AVAILABLE and len(image.shape) == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        pil_image = Image.fromarray(rgb_image)
        
        # Process image
        pixel_values = self.processor(
            pil_image,
            return_tensors="pt"
        ).pixel_values
        pixel_values = pixel_values.to(self.device)
        
        # Prepare decoder input
        decoder_input_ids = self.processor.tokenizer(
            task_prompt,
            add_special_tokens=False,
            return_tensors="pt"
        ).input_ids
        decoder_input_ids = decoder_input_ids.to(self.device)
        
        # Generate output
        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=self.max_length,
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                output_scores=True,
                return_dict_in_generate=True
            )
        
        # Decode output
        sequence = self.processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(self.processor.tokenizer.eos_token, "")
        sequence = sequence.replace(self.processor.tokenizer.pad_token, "")
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
        
        # Parse JSON from output
        extracted = self._parse_output(sequence)
        
        # Calculate confidence from generation scores
        if hasattr(outputs, 'scores') and outputs.scores:
            scores = torch.stack(outputs.scores, dim=1)
            probs = torch.softmax(scores, dim=-1)
            max_probs = probs.max(dim=-1).values
            confidence = float(max_probs.mean().cpu())
        else:
            confidence = 0.7  # Default confidence
        
        return extracted, confidence
    
    def _parse_output(self, output: str) -> Dict[str, Any]:
        """
        Parse Donut output into structured dictionary.
        
        Handles various output formats including JSON-like structures
        and key-value patterns.
        """
        # Try to extract JSON
        try:
            # Find JSON-like content
            json_match = re.search(r'\{.*\}', output, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
        
        # Try XML-like tag parsing (Donut often outputs this)
        result = {}
        tag_pattern = r'<s_(\w+)>(.*?)</s_\1>'
        
        for match in re.finditer(tag_pattern, output, re.DOTALL):
            key = match.group(1)
            value = match.group(2).strip()
            
            # Handle nested structures
            if '<' in value:
                # Recursively parse nested tags
                nested = self._parse_output(value)
                if nested:
                    result[key] = nested
                else:
                    result[key] = value
            else:
                result[key] = value
        
        # If no tags found, try simple key: value parsing
        if not result:
            for line in output.split('\n'):
                if ':' in line:
                    parts = line.split(':', 1)
                    key = parts[0].strip().lower().replace(' ', '_')
                    value = parts[1].strip()
                    if key and value:
                        result[key] = value
        
        return result
    
    def extract_to_fields(
        self,
        image: np.ndarray,
        page_num: int = 1,
        task_prompt: str = "<s_cord-v2>"
    ) -> List[Field]:
        """
        Extract and convert to Field objects.
        
        Args:
            image: Document image
            page_num: Page number for field metadata
            task_prompt: Task prompt for Donut
            
        Returns:
            List of extracted fields
        """
        extracted, confidence = self.extract(image, task_prompt)
        
        fields = self._dict_to_fields(extracted, confidence, page_num)
        
        return fields
    
    def _dict_to_fields(
        self,
        data: Dict[str, Any],
        confidence: float,
        page_num: int,
        prefix: str = ""
    ) -> List[Field]:
        """
        Convert extracted dictionary to Field objects.
        
        Recursively handles nested structures.
        """
        fields = []
        
        for key, value in data.items():
            field_name = f"{prefix}{key}" if prefix else key
            
            if isinstance(value, dict):
                # Recursive handling of nested dicts
                nested_fields = self._dict_to_fields(
                    value, confidence, page_num, f"{field_name}."
                )
                fields.extend(nested_fields)
            elif isinstance(value, list):
                # Handle lists (e.g., line items)
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        nested_fields = self._dict_to_fields(
                            item, confidence, page_num, f"{field_name}[{i}]."
                        )
                        fields.extend(nested_fields)
                    else:
                        fields.append(self._create_field(
                            f"{field_name}[{i}]", item, confidence, page_num
                        ))
            else:
                fields.append(self._create_field(
                    field_name, value, confidence, page_num
                ))
        
        return fields
    
    def _create_field(
        self,
        name: str,
        value: Any,
        confidence: float,
        page_num: int
    ) -> Field:
        """Create a Field object from extracted data."""
        # Determine data type
        data_type = "string"
        if isinstance(value, (int, float)):
            data_type = "number"
        elif self._looks_like_date(str(value)):
            data_type = "date"
        elif self._looks_like_currency(str(value)):
            data_type = "currency"
        
        # Create candidate
        candidate = Candidate(
            source=SourceEngine.DONUT,
            value=value,
            confidence=confidence,
            page=page_num
        )
        
        # Determine status based on confidence
        if confidence >= 0.8:
            status = FieldStatus.CONFIDENT
        elif confidence >= 0.5:
            status = FieldStatus.SINGLE_SOURCE
        else:
            status = FieldStatus.UNCERTAIN
        
        return Field(
            name=name,
            value=value,
            data_type=data_type,
            confidence=confidence,
            status=status,
            page=page_num,
            chosen_source=SourceEngine.DONUT,
            candidates=[candidate]
        )
    
    def _looks_like_date(self, value: str) -> bool:
        """Check if value looks like a date."""
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # 2024-01-15
            r'\d{2}/\d{2}/\d{4}',  # 01/15/2024
            r'\d{2}-\d{2}-\d{4}',  # 15-01-2024
            r'\w+ \d{1,2}, \d{4}',  # January 15, 2024
        ]
        return any(re.search(p, value) for p in date_patterns)
    
    def _looks_like_currency(self, value: str) -> bool:
        """Check if value looks like a currency amount."""
        currency_patterns = [
            r'[$€£¥]\s*[\d,]+\.?\d*',  # $1,234.56
            r'[\d,]+\.?\d*\s*[$€£¥]',  # 1,234.56 USD
            r'\d+[.,]\d{2}',  # 1234.56 or 1234,56
        ]
        return any(re.search(p, value) for p in currency_patterns)
    
    def get_supported_tasks(self) -> List[str]:
        """
        Get list of supported task prompts.
        
        Different Donut models support different task prompts.
        """
        # Common task prompts for different fine-tuned models
        return [
            "<s_cord-v2>",  # CORD (receipt parsing)
            "<s_docvqa>",  # Document VQA
            "<s_synthdog>",  # Synthetic document generation
            "<s>",  # Generic
        ]
