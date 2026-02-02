"""Decision engine for Accept/Review/Reject classification."""

from typing import Dict, List
from enum import Enum
from dataclasses import dataclass
from .confidence import DocumentConfidence


class Decision(Enum):
    """Document processing decision."""
    ACCEPT = "accept"
    REVIEW = "review"
    REJECT = "reject"


@dataclass
class DecisionResult:
    """Result of decision engine."""
    decision: Decision
    confidence_score: float
    reasons: List[str]
    hard_rejection: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'decision': self.decision.value,
            'confidence_score': self.confidence_score,
            'reasons': self.reasons,
            'hard_rejection': self.hard_rejection
        }


class DecisionEngine:
    """Makes Accept/Review/Reject decisions based on confidence scores."""
    
    def __init__(self, config: Dict):
        """Initialize decision engine.
        
        Args:
            config: Decision configuration
        """
        self.config = config
        self.accept_threshold = config.get('accept_threshold', 0.85)
        self.review_threshold = config.get('review_threshold', 0.60)
        
        # Hard rejection rules
        hard_reject_config = config.get('hard_reject', {})
        self.reject_no_text = hard_reject_config.get('no_text_detected', True)
        self.reject_quality_fail = hard_reject_config.get('quality_gate_failed', True)
        self.reject_missing_mandatory = hard_reject_config.get('mandatory_field_missing', True)
        self.max_non_alphanumeric = hard_reject_config.get('excessive_non_alphanumeric', 0.7)
    
    def make_decision(self,
                     document_confidence: DocumentConfidence,
                     quality_passed: bool,
                     text_detected: bool,
                     mandatory_fields_present: bool,
                     non_alphanumeric_ratio: float = 0.0) -> DecisionResult:
        """Make processing decision for document.
        
        Args:
            document_confidence: Document confidence scores
            quality_passed: Whether image quality gate passed
            text_detected: Whether any text was detected
            mandatory_fields_present: Whether all mandatory fields are present
            non_alphanumeric_ratio: Ratio of non-alphanumeric characters
            
        Returns:
            DecisionResult object
        """
        reasons = []
        hard_rejection = False
        
        # Check hard rejection rules first
        if self.reject_no_text and not text_detected:
            reasons.append("No text detected in image")
            hard_rejection = True
            return DecisionResult(
                decision=Decision.REJECT,
                confidence_score=0.0,
                reasons=reasons,
                hard_rejection=True
            )
        
        if self.reject_quality_fail and not quality_passed:
            reasons.append("Image failed quality gate")
            hard_rejection = True
            return DecisionResult(
                decision=Decision.REJECT,
                confidence_score=document_confidence.final_score,
                reasons=reasons,
                hard_rejection=True
            )
        
        if self.reject_missing_mandatory and not mandatory_fields_present:
            reasons.append("Mandatory fields are missing")
            hard_rejection = True
            return DecisionResult(
                decision=Decision.REJECT,
                confidence_score=document_confidence.final_score,
                reasons=reasons,
                hard_rejection=True
            )
        
        if non_alphanumeric_ratio > self.max_non_alphanumeric:
            reasons.append(f"Excessive non-alphanumeric content ({non_alphanumeric_ratio:.1%})")
            hard_rejection = True
            return DecisionResult(
                decision=Decision.REJECT,
                confidence_score=document_confidence.final_score,
                reasons=reasons,
                hard_rejection=True
            )
        
        # Make decision based on confidence score
        score = document_confidence.final_score
        
        if score >= self.accept_threshold:
            decision = Decision.ACCEPT
            reasons.append(f"Confidence score {score:.2f} exceeds accept threshold {self.accept_threshold}")
            
            # Add positive indicators
            if document_confidence.image_quality_score > 0.8:
                reasons.append("High image quality")
            if document_confidence.semantic_score > 0.9:
                reasons.append("Strong semantic validation")
            if document_confidence.consistency_score > 0.9:
                reasons.append("Excellent consistency checks")
        
        elif score >= self.review_threshold:
            decision = Decision.REVIEW
            reasons.append(f"Confidence score {score:.2f} in review range [{self.review_threshold}, {self.accept_threshold})")
            
            # Add specific concerns
            if document_confidence.image_quality_score < 0.6:
                reasons.append("Low image quality")
            if document_confidence.ocr_confidence_score < 0.7:
                reasons.append("Low OCR confidence")
            if document_confidence.semantic_score < 0.7:
                reasons.append("Weak semantic validation")
            if document_confidence.layout_score < 0.5:
                reasons.append("Poor layout match")
            if document_confidence.consistency_score < 0.7:
                reasons.append("Consistency issues detected")
        
        else:
            decision = Decision.REJECT
            reasons.append(f"Confidence score {score:.2f} below review threshold {self.review_threshold}")
            
            # Add specific failure reasons
            if document_confidence.image_quality_score < 0.5:
                reasons.append("Very poor image quality")
            if document_confidence.ocr_confidence_score < 0.5:
                reasons.append("Very low OCR confidence")
            if document_confidence.semantic_score < 0.5:
                reasons.append("Failed semantic validation")
            if document_confidence.layout_score < 0.3:
                reasons.append("Layout does not match expected template")
            if document_confidence.consistency_score < 0.5:
                reasons.append("Failed consistency checks")
        
        return DecisionResult(
            decision=decision,
            confidence_score=score,
            reasons=reasons,
            hard_rejection=hard_rejection
        )
    
    def get_threshold_info(self) -> Dict:
        """Get information about decision thresholds.
        
        Returns:
            Dictionary with threshold information
        """
        return {
            'accept_threshold': self.accept_threshold,
            'review_threshold': self.review_threshold,
            'reject_threshold': self.review_threshold,
            'ranges': {
                'accept': f'>= {self.accept_threshold}',
                'review': f'[{self.review_threshold}, {self.accept_threshold})',
                'reject': f'< {self.review_threshold}'
            }
        }
