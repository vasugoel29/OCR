"""Multi-stage confidence scoring module (Approach 6)."""

from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class FieldConfidence:
    """Confidence score for a single field."""
    field_name: str
    ocr_confidence: float
    semantic_validity: float
    positional_validity: float
    composite_score: float


@dataclass
class DocumentConfidence:
    """Overall document confidence scores."""
    image_quality_score: float
    ocr_confidence_score: float
    semantic_score: float
    layout_score: float
    consistency_score: float
    final_score: float
    field_confidences: Dict[str, FieldConfidence] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'image_quality_score': self.image_quality_score,
            'ocr_confidence_score': self.ocr_confidence_score,
            'semantic_score': self.semantic_score,
            'layout_score': self.layout_score,
            'consistency_score': self.consistency_score,
            'final_score': self.final_score,
            'field_confidences': {
                name: {
                    'field_name': fc.field_name,
                    'ocr_confidence': fc.ocr_confidence,
                    'semantic_validity': fc.semantic_validity,
                    'positional_validity': fc.positional_validity,
                    'composite_score': fc.composite_score
                }
                for name, fc in self.field_confidences.items()
            }
        }


class ConfidenceScorer:
    """Calculates multi-stage confidence scores."""
    
    def __init__(self, config: Dict):
        """Initialize confidence scorer.
        
        Args:
            config: Scoring configuration
        """
        self.config = config
        
        # Stage weights
        weights = config.get('weights', {})
        self.weight_image_quality = weights.get('image_quality', 0.20)
        self.weight_ocr_confidence = weights.get('ocr_confidence', 0.25)
        self.weight_semantic = weights.get('semantic_validity', 0.25)
        self.weight_layout = weights.get('layout_validity', 0.15)
        self.weight_consistency = weights.get('consistency', 0.15)
        
        # Field weights
        self.field_weights = config.get('field_weights', {})
    
    def calculate_document_confidence(self,
                                     image_quality_score: float,
                                     ocr_confidence_score: float,
                                     semantic_score: float,
                                     layout_score: float,
                                     consistency_score: float,
                                     field_scores: Optional[Dict[str, FieldConfidence]] = None) -> DocumentConfidence:
        """Calculate overall document confidence.
        
        Args:
            image_quality_score: Image quality score [0, 1]
            ocr_confidence_score: OCR confidence score [0, 1]
            semantic_score: Semantic validation score [0, 1]
            layout_score: Layout validation score [0, 1]
            consistency_score: Consistency validation score [0, 1]
            field_scores: Optional field-level confidence scores
            
        Returns:
            DocumentConfidence object
        """
        # Calculate weighted final score
        final_score = (
            self.weight_image_quality * image_quality_score +
            self.weight_ocr_confidence * ocr_confidence_score +
            self.weight_semantic * semantic_score +
            self.weight_layout * layout_score +
            self.weight_consistency * consistency_score
        )
        
        # Ensure score is in [0, 1]
        final_score = max(0.0, min(1.0, final_score))
        
        return DocumentConfidence(
            image_quality_score=image_quality_score,
            ocr_confidence_score=ocr_confidence_score,
            semantic_score=semantic_score,
            layout_score=layout_score,
            consistency_score=consistency_score,
            final_score=final_score,
            field_confidences=field_scores or {}
        )
    
    def calculate_field_confidence(self,
                                   field_name: str,
                                   ocr_confidence: float,
                                   semantic_valid: bool,
                                   positional_valid: bool = True) -> FieldConfidence:
        """Calculate confidence for a single field.
        
        Args:
            field_name: Name of the field
            ocr_confidence: OCR confidence for this field [0, 100]
            semantic_valid: Whether field passed semantic validation
            positional_valid: Whether field is in expected position
            
        Returns:
            FieldConfidence object
        """
        # Normalize OCR confidence to [0, 1]
        ocr_score = ocr_confidence / 100.0
        
        # Convert boolean validations to scores
        semantic_score = 1.0 if semantic_valid else 0.0
        positional_score = 1.0 if positional_valid else 0.5  # Partial credit
        
        # Calculate composite score
        composite = (
            0.4 * ocr_score +
            0.4 * semantic_score +
            0.2 * positional_score
        )
        
        return FieldConfidence(
            field_name=field_name,
            ocr_confidence=ocr_score,
            semantic_validity=semantic_score,
            positional_validity=positional_score,
            composite_score=composite
        )
    
    def calculate_weighted_field_score(self,
                                       field_confidences: Dict[str, FieldConfidence],
                                       document_type: str) -> float:
        """Calculate weighted average of field confidence scores.
        
        Args:
            field_confidences: Dictionary of field confidences
            document_type: Type of document ('invoice' or 'id_document')
            
        Returns:
            Weighted average score [0, 1]
        """
        if not field_confidences:
            return 0.0
        
        # Get field weights for this document type
        type_weights = self.field_weights.get(document_type, {})
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for field_name, field_conf in field_confidences.items():
            weight = type_weights.get(field_name, 1.0)
            weighted_sum += field_conf.composite_score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight
