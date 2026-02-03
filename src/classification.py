"""Document classification logic."""

from typing import Dict, List, Tuple
import re
import logging

logger = logging.getLogger('ocr_pipeline')

class DocumentClassifier:
    """Classifies documents based on content."""
    
    def __init__(self):
        """Initialize classifier with keyword maps and patterns."""
        self.type_keywords = {
            'id_document': [
                'government of india', 'male', 'female', 'dob', 'date of birth',
                'aadhaar', 'permanent account number', 'income tax department',
                'unique identification authority', 'gender', 'father', 'husband',
                'yob', 'year of birth', 'enrollment', 'resident', 'identity', 'card'
            ],
            'invoice': [
                'invoice', 'total', 'tax', 'gst', 'grand total', 'subtotal',
                'bill to', 'ship to', 'order date', 'payment', 'amount', 'qty',
                'description', 'hsn', 'sac', 'price', 'rate', 'due date'
            ]
        }
        
        # Regex patterns for strong signals
        self.type_patterns = {
            'id_document': [
                r'\d{4}\s\d{4}\s\d{4}',  # Aadhaar (12 digits space sep)
                r'[A-Z]{5}[0-9]{4}[A-Z]{1}',  # PAN
                r'DOB\s*:\s*\d{2}/\d{2}/\d{4}'  # DOB label
            ],
            'invoice': [
                r'Invoice\s*(No|#)',
                r'Total\s*(Amount)?'
            ]
        }
        
    def classify(self, text: str) -> str:
        """Classify document based on text content.
        
        Args:
            text: OCR extracted text
            
        Returns:
            Computed document type ('invoice' or 'id_document')
        """
        text = text.lower()
        scores = {dtype: 0 for dtype in self.type_keywords}
        
        logger.debug(f"Classifying text (len={len(text)}): {text[:100]}...")
        
        # Keyword scoring
        for dtype, keywords in self.type_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    # Longer keywords are stronger indicators
                    weight = 2 if len(keyword.split()) > 1 else 1
                    scores[dtype] += weight
                    
        # Regex scoring (Strong signals)
        for dtype, patterns in self.type_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    scores[dtype] += 5  # High weight for pattern match
                    logger.debug(f"Matched pattern for {dtype}: {pattern}")
        
        logger.info(f"Classification scores: {scores}")
        
        # Default logic
        # If ID score > 0, prefer ID (since they often have less text than invoices)
        if scores['id_document'] > 0 and scores['id_document'] >= scores['invoice']:
            return 'id_document'
            
        if scores['invoice'] > scores['id_document']:
            return 'invoice'
            
        # Tie-breaker or both zero
        # If short text (< 50 words) and mostly numbers, might be an ID card back or crop
        # But risky. Defaulting to invoice is safer for business apps, 
        # but let's check if we have ANY signal.
        
        if scores['id_document'] == 0 and scores['invoice'] == 0:
             # Check numeric density?
             # For now, default invoice
             return 'invoice'
             
        return 'invoice'
