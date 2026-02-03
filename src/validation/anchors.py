"""Fuzzy anchor detection logic."""

from typing import Dict, List, Tuple
from rapidfuzz import process, fuzz
from ..ocr.models import OCRResult

class AnchorValidator:
    """Detects fuzzy anchors in document text."""
    
    def __init__(self, config: Dict):
        """Initialize anchor validator.
        
        Args:
            config: Validation configuration containing 'anchors' section
        """
        self.config = config.get('anchors', {})
        
    def validate_anchors(self, text: str, document_type: str) -> Tuple[float, Dict[str, float]]:
        """Validate presence of anchor keywords.
        
        Args:
            text: Full text content
            document_type: Type of document
            
        Returns:
            Tuple of (FuzzyScore, details_dict)
        """
        if document_type not in self.config:
            return 0.0, {'error': f'No anchor config for {document_type}'}
            
        doc_config = self.config[document_type]
        required_anchors = doc_config.get('required', [])
        optional_anchors = doc_config.get('optional', [])
        threshold = doc_config.get('threshold', 80)
        
        # Split text into lines/words for better matching against short keywords
        # But rapidfuzz partial_ratio works well on full text too
        text_lower = text.lower()
        
        found_required = 0
        found_optional = 0
        matches = {}
        
        # Check required anchors
        for anchor in required_anchors:
            # simple substring check first (fast)
            if anchor in text_lower:
                matches[anchor] = 100.0
                found_required += 1
                continue
                
            # Fuzzy check
            # partial_token_ratio usually works well for finding "total" in "Sub Total: $100"
            score = fuzz.partial_token_sort_ratio(anchor, text_lower)
            if score >= threshold:
                matches[anchor] = score
                found_required += 1
            else:
                matches[anchor] = score
        
        # Check optional anchors
        for anchor in optional_anchors:
            if anchor in text_lower:
                matches[anchor] = 100.0
                found_optional += 1
                continue
                
            score = fuzz.partial_token_sort_ratio(anchor, text_lower)
            if score >= threshold:
                matches[anchor] = score
                found_optional += 1
                
        total_required = len(required_anchors)
        
        # FuzzyScore logic:
        # Base score on required anchors found
        if total_required > 0:
            required_ratio = found_required / total_required
        else:
            required_ratio = 1.0
            
        # Bonus for optional anchors (capped)
        optional_bonus = min(0.2, found_optional * 0.05)
        
        final_score = min(1.0, required_ratio + optional_bonus)
        
        details = {
            'found_required': found_required,
            'total_required': total_required,
            'found_optional': found_optional,
            'matches': matches
        }
        
        return final_score, details
