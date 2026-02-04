"""Normalization utilities for multilingual text processing."""

import re
from typing import Optional

def convert_devanagari_to_arabic(text: str) -> str:
    """Convert Devanagari numerals to Arabic digits.
    
    Args:
        text: Input text containing Devanagari digits (०-९).
        
    Returns:
        Text with Arabic digits (0-9).
    """
    devanagari_digits = str.maketrans("०१२३४५६७८९", "0123456789")
    return text.translate(devanagari_digits)

def normalize_date(date_str: str) -> Optional[str]:
    """Normalize date string to DD/MM/YYYY format.
    
    Handles:
    - DD-MM-YYYY
    - DD.MM.YYYY
    - DD/MM/YYYY
    - YYYY-MM-DD
    
    Args:
        date_str: Raw date string.
        
    Returns:
        Normalized date string or None if invalid.
    """
    if not date_str:
        return None
        
    # Clean string
    clean_date = re.sub(r'[^\d/\-\.]', '', date_str)
    
    # Try different formats
    # DD/MM/YYYY or DD-MM-YYYY or DD.MM.YYYY
    dmy_match = re.match(r'^(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})$', clean_date)
    if dmy_match:
        d, m, y = dmy_match.groups()
        return f"{int(d):02d}/{int(m):02d}/{y}"
        
    # YYYY-MM-DD or YYYY/MM/DD
    ymd_match = re.match(r'^(\d{4})[/\-\.](\d{1,2})[/\-\.](\d{1,2})$', clean_date)
    if ymd_match:
        y, m, d = ymd_match.groups()
        return f"{int(d):02d}/{int(m):02d}/{y}"
        
    return None
