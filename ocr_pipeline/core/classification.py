"""Document classification logic for Indian documents."""

from typing import Dict, List, Tuple
import re
import logging

logger = logging.getLogger('ocr_pipeline')

class DocumentClassifier:
    """Classifies Indian documents (Aadhaar, PAN, Vehicle RC) based on content."""
    
    def __init__(self):
        """Initialize classifier with keyword maps and patterns."""
        self.type_keywords = {
            'aadhaar': [
                'aadhaar', 'आधार', 'uidai', 'government of india',
                'भारत सरकार', 'unique identification', 'unique identification authority',
                'enrollment', 'resident', 'dob', 'date of birth', 'male', 'female',
                'gender', 'address', 'पता'
            ],
            'pan': [
                'income tax', 'permanent account number', 'pan',
                'income tax department', 'govt. of india', 'government of india',
                'आयकर विभाग', 'स्थायी खाता संख्या', 'father', 'signature',
                'fathers name', 'father\'s name'
            ],
            'vehicle_rc': [
                'registration certificate', 'vehicle', 'registration number',
                'engine no', 'chassis no', 'registering authority', 'owner',
                'रजिस्ट्रेशन', 'वाहन', 'इंजन', 'चेसिस', 'maker', 'model',
                'vehicle class', 'reg no', 'rc', 'rto'
            ]
        }
        
        # Regex patterns for strong signals
        self.type_patterns = {
            'aadhaar': [
                r'\b\d{4}\s+\d{4}\s+\d{4}\b',  # Aadhaar number (12 digits with spaces)
                r'\b\d{12}\b',  # Aadhaar number (12 digits continuous)
                r'(?:aadhaar|आधार)',  # Aadhaar keyword
                r'UIDAI',  # UIDAI keyword
            ],
            'pan': [
                r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b',  # PAN format
                # Fuzzy matches for "Income Tax Department"
                r'[I1|]NCOME\s*TAX\s*DEP[A-Z]*',
                r'NCOME\s*T[A-X]+',  # Handle missing leading I
                # Fuzzy matches for "Permanent Account Number"
                r'P[AE]RM[A-Z]*\s*ACC[A-Z]*\s*NUM[A-Z]*',
                r'(?:father\'?s?\s+name)',  # Father's name
                r'GOVT\.?\s*O[Ff]\s*IND[A-Z]*', # Govt of India
            ],
            'vehicle_rc': [
                r'\b[A-Z]{2}\s*[-]?\s*\d{2}\s*[-]?\s*[A-Z]{1,2}\s*[-]?\s*\d{4}\b',  # Registration number
                r'(?:registration\s+certificate|vehicle\s+informa)',
                r'(?:chassis|engine\s+no)',
                r'(?:fuel|seating|unladen|wheel\s*base)',
                r'(?:mfg\s*date|form\s+23)',
                r'(?:model|maker|manufacturer)',
            ],
        }
    
    def classify_with_scores(self, text: str) -> Tuple[str, Dict[str, int]]:
        """Classify document and return scores.
        
        Args:
            text: OCR extracted text
            
        Returns:
            Tuple of (best_type, scores_dict)
        """
        text_lower = text.lower()
        scores = {dtype: 0 for dtype in self.type_keywords}
        
        logger.debug(f"Classifying text (len={len(text)}): {text[:100]}...")
        
        # Keyword scoring
        for dtype, keywords in self.type_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    # Longer keywords are stronger indicators
                    weight = 2 if len(keyword.split()) > 1 else 1
                    scores[dtype] += weight
        
        # Regex scoring (Strong signals)
        for dtype, patterns in self.type_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    scores[dtype] += 5  # High weight for pattern match
        
        logger.info(f"Classification scores: {scores}")
        
        # Decision logic - highest score wins
        max_score = max(scores.values())
        
        # If all scores are 0, default to aadhaar (most common)
        if max_score == 0:
            logger.warning("No classification signals found, defaulting to 'aadhaar'")
            return 'aadhaar', scores
        
        # Get document type with highest score
        classified_type = max(scores, key=scores.get)
        
        # Tie-breaker: if multiple types have same score, use priority order
        # Priority: vehicle_rc > pan > aadhaar (more specific to less specific)
        if list(scores.values()).count(max_score) > 1:
            logger.info(f"Tie detected, using priority order")
            priority_order = ['vehicle_rc', 'pan', 'aadhaar']
            for dtype in priority_order:
                if scores[dtype] == max_score:
                    classified_type = dtype
                    break
        
        logger.info(f"Classified as: {classified_type} (score: {scores[classified_type]})")
        return classified_type, scores

    def classify(self, text: str) -> str:
        """Classify document based on text content (Legacy wrapper)."""
        dtype, _ = self.classify_with_scores(text)
        return dtype

