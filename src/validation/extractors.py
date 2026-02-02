"""Field extraction logic using pattern matching."""

import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from ..ocr.models import OCRResult


class FieldExtractor:
    """Extracts structured fields from OCR text."""
    
    def __init__(self, config: Dict):
        """Initialize field extractor.
        
        Args:
            config: Semantic validation configuration
        """
        self.config = config
    
    def extract_invoice_fields(self, ocr_result: OCRResult) -> Dict[str, any]:
        """Extract fields from invoice OCR result.
        
        Args:
            ocr_result: OCR result object
            
        Returns:
            Dictionary of extracted fields
        """
        text = ocr_result.full_text
        fields = {}
        
        # Extract invoice number
        invoice_number = self._extract_invoice_number(text)
        if invoice_number:
            fields['invoice_number'] = invoice_number
        
        # Extract dates
        dates = self._extract_dates(text)
        if dates:
            fields['date'] = dates[0]  # First date is usually invoice date
            if len(dates) > 1:
                fields['due_date'] = dates[1]
        
        # Extract amounts
        amounts = self._extract_amounts(text)
        if amounts:
            # Try to identify total, subtotal, tax
            fields.update(self._identify_amount_types(text, amounts))
        
        # Extract vendor name (usually near top)
        vendor = self._extract_vendor_name(ocr_result)
        if vendor:
            fields['vendor_name'] = vendor
        
        return fields
    
    def extract_id_fields(self, ocr_result: OCRResult) -> Dict[str, any]:
        """Extract fields from ID document OCR result.
        
        Args:
            ocr_result: OCR result object
            
        Returns:
            Dictionary of extracted fields
        """
        text = ocr_result.full_text
        fields = {}
        
        # Extract ID number
        id_number = self._extract_id_number(text)
        if id_number:
            fields['id_number'] = id_number
        
        # Extract name
        name = self._extract_name(text)
        if name:
            fields['name'] = name
        
        # Extract dates
        dates = self._extract_dates(text)
        if dates:
            # Try to identify DOB, issue date, expiry date
            fields.update(self._identify_date_types(text, dates))
        
        # Extract address (if present)
        address = self._extract_address(text)
        if address:
            fields['address'] = address
        
        return fields
    
    def _extract_invoice_number(self, text: str) -> Optional[str]:
        """Extract invoice number from text."""
        patterns = [
            r'invoice\s*#?\s*:?\s*([A-Z0-9-]{6,20})',
            r'inv\s*#?\s*:?\s*([A-Z0-9-]{6,20})',
            r'#\s*([A-Z0-9-]{6,20})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_dates(self, text: str) -> List[str]:
        """Extract all dates from text."""
        patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',
            r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}',
        ]
        
        dates = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dates.extend(matches)
        
        return dates
    
    def _extract_amounts(self, text: str) -> List[float]:
        """Extract monetary amounts from text."""
        # Pattern for currency amounts
        pattern = r'[$€£¥]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        
        matches = re.findall(pattern, text)
        amounts = []
        
        for match in matches:
            try:
                # Remove commas and convert to float
                amount = float(match.replace(',', ''))
                amounts.append(amount)
            except ValueError:
                continue
        
        return amounts
    
    def _identify_amount_types(self, text: str, amounts: List[float]) -> Dict[str, float]:
        """Identify which amounts are total, subtotal, tax."""
        fields = {}
        text_lower = text.lower()
        
        # Find total
        total_pattern = r'total\s*:?\s*[$€£¥]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        total_match = re.search(total_pattern, text_lower)
        if total_match:
            try:
                fields['total_amount'] = float(total_match.group(1).replace(',', ''))
            except ValueError:
                pass
        elif amounts:
            # Assume largest amount is total
            fields['total_amount'] = max(amounts)
        
        # Find subtotal
        subtotal_pattern = r'subtotal\s*:?\s*[$€£¥]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        subtotal_match = re.search(subtotal_pattern, text_lower)
        if subtotal_match:
            try:
                fields['subtotal'] = float(subtotal_match.group(1).replace(',', ''))
            except ValueError:
                pass
        
        # Find tax
        tax_pattern = r'tax\s*:?\s*[$€£¥]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        tax_match = re.search(tax_pattern, text_lower)
        if tax_match:
            try:
                fields['tax'] = float(tax_match.group(1).replace(',', ''))
            except ValueError:
                pass
        
        return fields
    
    def _extract_vendor_name(self, ocr_result: OCRResult) -> Optional[str]:
        """Extract vendor name (usually in top portion)."""
        if not ocr_result.lines:
            return None
        
        # Get first few lines (vendor name usually at top)
        top_lines = ocr_result.lines[:3]
        
        # Find longest line with good confidence
        best_line = None
        max_length = 0
        
        for line in top_lines:
            if line.confidence > 60 and len(line.text) > max_length:
                # Skip lines that look like addresses or dates
                if not re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', line.text):
                    best_line = line.text
                    max_length = len(line.text)
        
        return best_line
    
    def _extract_id_number(self, text: str) -> Optional[str]:
        """Extract ID number from text (including Aadhaar)."""
        # Aadhaar number: 12 digits, often with spaces (XXXX XXXX XXXX)
        aadhaar_patterns = [
            r'(\d{4}\s*\d{4}\s*\d{4})',  # Aadhaar with spaces
            r'(\d{12})',  # Aadhaar without spaces
        ]
        
        for pattern in aadhaar_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # Clean and validate
                clean_number = re.sub(r'\s+', '', match)
                if len(clean_number) == 12 and clean_number.isdigit():
                    return clean_number
        
        # Other ID patterns
        patterns = [
            r'(?:id|number|vid)\s*:?\s*([A-Z0-9\s]{8,20})',
            r'([A-Z]{2}\d{6,12})',  # Common ID format (PAN, etc.)
            r'([A-Z0-9]{8,15})',  # Generic alphanumeric ID
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_name(self, text: str) -> Optional[str]:
        """Extract person name from text."""
        # Look for name pattern (capitalized words)
        pattern = r'name\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)'
        match = re.search(pattern, text)
        
        if match:
            return match.group(1)
        
        # Alternative: look for capitalized name pattern
        pattern = r'([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
        matches = re.findall(pattern, text)
        
        if matches:
            # Return first match that looks like a name
            for match in matches:
                if len(match.split()) >= 2:
                    return match
        
        return None
    
    def _identify_date_types(self, text: str, dates: List[str]) -> Dict[str, str]:
        """Identify which dates are DOB, issue, expiry."""
        fields = {}
        text_lower = text.lower()
        
        # Find date of birth
        dob_pattern = r'(?:date of birth|dob|born)\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
        dob_match = re.search(dob_pattern, text_lower)
        if dob_match:
            fields['date_of_birth'] = dob_match.group(1)
        
        # Find issue date
        issue_pattern = r'(?:issue|issued)\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
        issue_match = re.search(issue_pattern, text_lower)
        if issue_match:
            fields['issue_date'] = issue_match.group(1)
        
        # Find expiry date
        expiry_pattern = r'(?:expir|valid until)\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
        expiry_match = re.search(expiry_pattern, text_lower)
        if expiry_match:
            fields['expiry_date'] = expiry_match.group(1)
        
        return fields
    
    def _extract_address(self, text: str) -> Optional[str]:
        """Extract address from text."""
        # Look for address pattern (street number + street name)
        pattern = r'\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln))'
        match = re.search(pattern, text)
        
        if match:
            return match.group(0)
        
        return None
