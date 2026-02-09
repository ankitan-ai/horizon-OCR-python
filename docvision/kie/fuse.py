"""
Rank-and-fuse logic for combining multiple extraction sources.

Combines candidates from Donut, LayoutLMv3, and OCR engines
using weighted scoring and validator feedback.
"""

from typing import List, Dict, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict
from loguru import logger

from docvision.types import (
    Field, Candidate, SourceEngine, FieldStatus,
    BoundingBox, ValidatorResult
)


class FusionStrategy(Enum):
    """Strategy for fusing multiple candidates."""
    HIGHEST_CONFIDENCE = "highest_confidence"
    WEIGHTED_VOTE = "weighted_vote"
    VALIDATOR_PRIORITY = "validator_priority"
    CONSENSUS = "consensus"


@dataclass
class SourceWeight:
    """Weight configuration for a source."""
    source: SourceEngine
    weight: float = 1.0
    validator_bonus: float = 0.2  # Bonus for passing validators


class RankAndFuse:
    """
    Rank and fuse candidates from multiple extraction sources.
    
    Combines results from Donut, LayoutLMv3, and OCR using
    configurable strategies and validator feedback.
    """
    
    def __init__(
        self,
        strategy: FusionStrategy = FusionStrategy.WEIGHTED_VOTE,
        source_weights: Optional[Dict[SourceEngine, float]] = None,
        validator_bonus: float = 0.2,
        min_confidence: float = 0.2
    ):
        """
        Initialize rank-and-fuse engine.
        
        Args:
            strategy: Fusion strategy to use
            source_weights: Custom weights per source
            validator_bonus: Confidence bonus for passing validators
            min_confidence: Minimum confidence to include candidate
        """
        self.strategy = strategy
        self.validator_bonus = validator_bonus
        self.min_confidence = min_confidence
        
        # Default source weights
        self.source_weights = source_weights or {
            SourceEngine.DONUT: 1.0,
            SourceEngine.LAYOUTLMV3: 0.9,
            SourceEngine.TROCR: 0.8,
            SourceEngine.TESSERACT: 0.7,
            SourceEngine.PPSTRUCTURE: 0.85,
        }
    
    def fuse_fields(
        self,
        field_lists: List[List[Field]],
        validators: Optional[List[Callable]] = None
    ) -> List[Field]:
        """
        Fuse fields from multiple sources.
        
        Args:
            field_lists: List of field lists from different sources
            validators: Optional list of validator functions
            
        Returns:
            Fused list of fields with best values selected
        """
        # Group fields by name
        field_groups: Dict[str, List[Field]] = defaultdict(list)
        
        for fields in field_lists:
            for field in fields:
                # Normalize field name
                normalized_name = self._normalize_field_name(field.name)
                field_groups[normalized_name].append(field)
        
        # Fuse each field group
        fused_fields = []
        
        for field_name, fields in field_groups.items():
            fused = self._fuse_single_field(fields, validators)
            if fused:
                fused_fields.append(fused)
        
        return fused_fields
    
    def _normalize_field_name(self, name: str) -> str:
        """Normalize field name for matching."""
        return name.lower().strip().replace(" ", "_").replace("-", "_")
    
    def _fuse_single_field(
        self,
        fields: List[Field],
        validators: Optional[List[Callable]] = None
    ) -> Optional[Field]:
        """Fuse multiple field candidates into one."""
        if not fields:
            return None
        
        if len(fields) == 1:
            return fields[0]
        
        # Collect all candidates
        all_candidates = []
        for field in fields:
            all_candidates.extend(field.candidates)
            
            # Also add the field's main value as a candidate if not already present
            if field.chosen_source:
                main_candidate = Candidate(
                    source=field.chosen_source,
                    value=field.value,
                    confidence=field.confidence,
                    bbox=field.bbox,
                    page=field.page
                )
                # Avoid duplicates
                if not any(c.source == main_candidate.source and c.value == main_candidate.value 
                          for c in all_candidates):
                    all_candidates.append(main_candidate)
        
        if not all_candidates:
            return fields[0]  # Return first field if no candidates
        
        # Apply validators to candidates
        if validators:
            all_candidates = self._apply_validators(all_candidates, validators)
        
        # Select best candidate based on strategy
        if self.strategy == FusionStrategy.HIGHEST_CONFIDENCE:
            best = self._select_highest_confidence(all_candidates)
        elif self.strategy == FusionStrategy.WEIGHTED_VOTE:
            best = self._select_weighted_vote(all_candidates)
        elif self.strategy == FusionStrategy.VALIDATOR_PRIORITY:
            best = self._select_validator_priority(all_candidates)
        elif self.strategy == FusionStrategy.CONSENSUS:
            best = self._select_consensus(all_candidates)
        else:
            best = self._select_highest_confidence(all_candidates)
        
        if not best:
            return fields[0]
        
        # Create fused field
        fused = Field(
            name=fields[0].name,
            value=best.value,
            data_type=fields[0].data_type,
            confidence=best.confidence,
            status=self._determine_status(best, all_candidates),
            page=best.page or fields[0].page,
            bbox=best.bbox or fields[0].bbox,
            chosen_source=best.source,
            candidates=all_candidates,
            validators=[]
        )
        
        return fused
    
    def _apply_validators(
        self,
        candidates: List[Candidate],
        validators: List[Callable]
    ) -> List[Candidate]:
        """Apply validators and adjust confidence."""
        for candidate in candidates:
            validation_passed = 0
            validation_total = 0
            
            for validator in validators:
                try:
                    result = validator(candidate.value)
                    validation_total += 1
                    if result:
                        validation_passed += 1
                except Exception:
                    pass  # Skip failed validators
            
            # Apply validator bonus
            if validation_total > 0:
                pass_ratio = validation_passed / validation_total
                candidate.confidence = min(
                    1.0,
                    candidate.confidence + (self.validator_bonus * pass_ratio)
                )
                candidate.metadata["validation_passed"] = validation_passed
                candidate.metadata["validation_total"] = validation_total
        
        return candidates
    
    def _select_highest_confidence(
        self,
        candidates: List[Candidate]
    ) -> Optional[Candidate]:
        """Select candidate with highest confidence."""
        if not candidates:
            return None
        
        # Filter by minimum confidence
        valid = [c for c in candidates if c.confidence >= self.min_confidence]
        
        if not valid:
            return max(candidates, key=lambda c: c.confidence)
        
        return max(valid, key=lambda c: c.confidence)
    
    def _select_weighted_vote(
        self,
        candidates: List[Candidate]
    ) -> Optional[Candidate]:
        """Select using weighted voting based on source weights."""
        if not candidates:
            return None
        
        # Group by value
        value_scores: Dict[str, float] = defaultdict(float)
        value_candidates: Dict[str, Candidate] = {}
        
        for candidate in candidates:
            value_key = str(candidate.value).strip().lower()
            source_weight = self.source_weights.get(candidate.source, 0.5)
            score = candidate.confidence * source_weight
            
            value_scores[value_key] += score
            
            # Keep the highest confidence candidate for each value
            if value_key not in value_candidates or \
               candidate.confidence > value_candidates[value_key].confidence:
                value_candidates[value_key] = candidate
        
        if not value_scores:
            return self._select_highest_confidence(candidates)
        
        # Select value with highest weighted score
        best_value = max(value_scores.keys(), key=lambda k: value_scores[k])
        
        return value_candidates.get(best_value)
    
    def _select_validator_priority(
        self,
        candidates: List[Candidate]
    ) -> Optional[Candidate]:
        """Prioritize candidates that passed validators."""
        if not candidates:
            return None
        
        # Sort by validation pass ratio, then confidence
        def sort_key(c: Candidate) -> tuple:
            passed = c.metadata.get("validation_passed", 0)
            total = c.metadata.get("validation_total", 0)
            ratio = passed / total if total > 0 else 0
            return (ratio, c.confidence)
        
        return max(candidates, key=sort_key)
    
    def _select_consensus(
        self,
        candidates: List[Candidate]
    ) -> Optional[Candidate]:
        """Select value that appears in multiple sources."""
        if not candidates:
            return None
        
        # Count occurrences of each value
        value_counts: Dict[str, int] = defaultdict(int)
        value_candidates: Dict[str, List[Candidate]] = defaultdict(list)
        
        for candidate in candidates:
            value_key = str(candidate.value).strip().lower()
            value_counts[value_key] += 1
            value_candidates[value_key].append(candidate)
        
        # Find value with most sources agreeing
        max_count = max(value_counts.values())
        
        if max_count > 1:
            # Consensus found
            for value_key, count in value_counts.items():
                if count == max_count:
                    # Return highest confidence candidate for this value
                    return max(
                        value_candidates[value_key],
                        key=lambda c: c.confidence
                    )
        
        # No consensus, fall back to weighted vote
        return self._select_weighted_vote(candidates)
    
    def _determine_status(
        self,
        selected: Candidate,
        all_candidates: List[Candidate]
    ) -> FieldStatus:
        """Determine field status based on selection."""
        # Count sources with similar values
        selected_value = str(selected.value).strip().lower()
        matching_sources = set()
        
        for c in all_candidates:
            if str(c.value).strip().lower() == selected_value:
                matching_sources.add(c.source)
        
        # Check validation status
        passed = selected.metadata.get("validation_passed", 0)
        total = selected.metadata.get("validation_total", 0)
        
        if total > 0 and passed == total:
            return FieldStatus.VALIDATED
        
        if total > 0 and passed < total:
            return FieldStatus.VALIDATION_FAILED
        
        if len(matching_sources) >= 2 and selected.confidence >= 0.7:
            return FieldStatus.CONFIDENT
        
        if len(matching_sources) == 1:
            return FieldStatus.SINGLE_SOURCE
        
        if selected.confidence < 0.5:
            return FieldStatus.UNCERTAIN
        
        return FieldStatus.CONFIDENT


def merge_ocr_and_kie_fields(
    ocr_text_lines: List["TextLine"],
    kie_fields: List[Field]
) -> List[Field]:
    """
    Merge OCR text with KIE field extractions.
    
    Associates KIE field values with their source text lines
    based on bounding box overlap.
    
    Args:
        ocr_text_lines: Text lines from OCR
        kie_fields: Fields from KIE models
        
    Returns:
        Updated fields with OCR context
    """
    from docvision.types import TextLine
    
    for field in kie_fields:
        if not field.bbox:
            continue
        
        # Find overlapping text lines
        matching_lines = []
        for line in ocr_text_lines:
            if _boxes_overlap(field.bbox, line.bbox):
                matching_lines.append(line)
        
        if matching_lines:
            # Add OCR candidate
            ocr_text = " ".join(line.text for line in matching_lines)
            avg_conf = sum(line.confidence for line in matching_lines) / len(matching_lines)
            
            ocr_candidate = Candidate(
                source=matching_lines[0].source,
                value=ocr_text,
                confidence=avg_conf,
                bbox=field.bbox
            )
            
            # Add if not duplicate
            if not any(c.source == ocr_candidate.source for c in field.candidates):
                field.candidates.append(ocr_candidate)
    
    return kie_fields


def _boxes_overlap(box1: BoundingBox, box2: BoundingBox, threshold: float = 0.3) -> bool:
    """Check if two boxes overlap significantly."""
    x1 = max(box1.x1, box2.x1)
    y1 = max(box1.y1, box2.y1)
    x2 = min(box1.x2, box2.x2)
    y2 = min(box1.y2, box2.y2)
    
    if x2 <= x1 or y2 <= y1:
        return False
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = box1.area
    area2 = box2.area
    
    # IoU-like metric
    overlap_ratio = intersection / min(area1, area2) if min(area1, area2) > 0 else 0
    
    return overlap_ratio >= threshold
