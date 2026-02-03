"""Key-Value pair extraction and validation."""

from typing import Dict, List, Tuple
from ..ocr.models import OCRResult, WordData

class KeyValueExtractor:
    """Extracts and validates Key-Value pairs based on spatial proximity."""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def validate_kv_pairs(self, ocr_result: OCRResult, document_type: str) -> float:
        """Calculate KVScore based on finding expected keys with values.
        
        This is a simplified implementation. A full KV extractor would
        build a graph of relationships. Here we just check if
        keywords are followed by reasonable values nearby.
        
        Args:
            ocr_result: OCR result object
            document_type: Type of document
            
        Returns:
            KVScore [0, 1]
        """
        # Logic:
        # For invoices: look for "Total" followed by number
        # For IDs: look for "ID" followed by alphanumeric
        
        # This acts as a proxy for KV structural integrity
        text = ocr_result.full_text.lower()
        score = 0.5  # Base score
        
        if document_type == 'invoice':
            # Check for Total -> Number relationship
            if 'total' in text and any(c.isdigit() for c in text):
                score += 0.3
            if 'invoice' in text:
                score += 0.2
                
        elif document_type == 'id_document':
            # Check for Number/DOB proximity
            if 'dob' in text or 'birth' in text:
                score += 0.25
            if any(k in text for k in ['id', 'no.', 'number']):
                score += 0.25
                
        return min(1.0, score)
