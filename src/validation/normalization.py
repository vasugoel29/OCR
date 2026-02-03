"""Token normalization and cleaning logic."""

import re
from typing import List, Optional

class TokenNormalizer:
    """Handles text cleaning, normalization, and standardization."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Basic text cleaning."""
        if not text:
            return ""
        # Remove excessive whitespace
        return " ".join(text.split())

    @staticmethod
    def normalize_numeric_field(text: str) -> str:
        """Convert common OCR confusions in numeric context (O->0, l->1, etc.)."""
        if not text:
            return ""
        
        # Replacement map for numeric contexts
        replacements = {
            'O': '0', 'o': '0',
            'I': '1', 'l': '1', 'i': '1',
            'S': '5', 's': '5',
            'B': '8',
            'G': '6',
            'Z': '2', 'z': '2'
        }
        
        normalized = text
        for char, repl in replacements.items():
            normalized = normalized.replace(char, repl)
            
        # Strip non-numeric-ish chars (keep dots, commas, dashes)
        normalized = re.sub(r'[^0-9.,-]', '', normalized)
        
        return normalized

    @staticmethod
    def standardize_date(date_str: str) -> Optional[str]:
        """Convert various date formats to YYYY-MM-DD."""
        if not date_str:
            return None
            
        # normalize separators
        clean_date = re.sub(r'[./]', '-', date_str)
        
        try:
            from datetime import datetime
            from dateutil import parser
            dt = parser.parse(clean_date)
            return dt.strftime('%Y-%m-%d')
        except:
            return None
