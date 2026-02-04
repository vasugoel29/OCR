"""Business rule validation logic for OCR fields."""

from typing import Dict, Tuple, List, Optional
from datetime import datetime, timedelta
import re
import logging

class BusinessRuleValidator:
    """Validates extracted fields against business rules."""
    
    def __init__(self, config: Dict):
        """Initialize validator with configuration.
        
        Args:
            config: Business rules configuration
        """
        self.config = config
        self.date_config = config.get('date_validation', {})
        self.amount_config = config.get('amount_validation', {})
        self.logger = logging.getLogger(__name__)
        
        # Date rules
        self.reject_future_dates = self.date_config.get('reject_future_dates', True)
        self.max_age_days = self.date_config.get('max_age_days', 365)
        
        # Amount rules
        self.min_amount = self.amount_config.get('min_amount', 0.0)
        self.max_amount = self.amount_config.get('max_amount', 1000000.0)
        
    def validate(self, extracted_fields: Dict, document_type: str) -> Tuple[bool, List[str]]:
        """Run all business rule validations.
        
        Args:
            extracted_fields: Dictionary of extracted fields
            document_type: Type of document
            
        Returns:
            Tuple of (passed, list of failure reasons)
        """
        reasons = []
        
        # 1. Date Validation
        # Look for typical date fields
        date_fields = ['date', 'invoice_date', 'bill_date', 'due_date']
        if document_type == 'id_document':
            date_fields = ['date_of_birth', 'dob', 'issue_date']
            
        for field in date_fields:
            if field in extracted_fields:
                valid, reason = self.validate_date(extracted_fields[field], field)
                if not valid:
                    reasons.append(reason)
                    
        # 2. Amount Validation
        amount_fields = ['total_amount', 'amount', 'grand_total', 'net_amount']
        for field in amount_fields:
            if field in extracted_fields:
                valid, reason = self.validate_amount(extracted_fields[field], field)
                if not valid:
                    reasons.append(reason)
                    
        return len(reasons) == 0, reasons
    
    def validate_date(self, date_val: str, field_name: str) -> Tuple[bool, Optional[str]]:
        """Validate a date string."""
        # Simple parser for standard formats (YYYY-MM-DD, DD/MM/YYYY)
        # In a real system, use robust parsing from src/utils.py or dateutil
        try:
            parsed_date = self._parse_date(date_val)
            if not parsed_date:
                return True, None # Skip if cannot parse (handled by format validation)
            
            now = datetime.now()
            
            # Context checks
            is_dob = field_name in ['date_of_birth', 'dob']
            is_expiry = 'expiry' in field_name or 'valid_until' in field_name
            
            # Logic based on field type
            if is_dob:
                # DOB: Must be in past, not unreasonably old (e.g. > 120 years)
                # Max age check does NOT apply to DOB
                if parsed_date > now:
                    return False, f"Future DOB detected: {date_val}"
                if parsed_date < now - timedelta(days=365*120):
                     return False, f"DOB unreasonably old: {date_val}"
                     
            elif is_expiry:
                # Expiry: Usually in future, but can be in past (expired document)
                # We don't hard reject expired docs here unless configured, but definitely allow future
                pass 
                
            else:
                # Invoice/Transaction/Issue Date logic
                # Only apply future rejection and max age to these fields
                if self.reject_future_dates and parsed_date > now + timedelta(days=1): # 1 day buffer
                    return False, f"Future date detected in {field_name}: {date_val}"
                
                if self.max_age_days > 0:
                    min_date = now - timedelta(days=self.max_age_days)
                    if parsed_date < min_date:
                        return False, f"Date too old in {field_name}: {date_val} (> {self.max_age_days} days)"
                    
            return True, None
            
        except Exception as e:
            self.logger.warning(f"Date validation error for {date_val}: {e}")
            return True, None

    def validate_amount(self, amount_val: any, field_name: str) -> Tuple[bool, Optional[str]]:
        """Validate a numeric amount."""
        try:
            # Handle string amounts with currency symbols
            if isinstance(amount_val, str):
                # Remove currency symbols and commas
                clean_val = re.sub(r'[^\d.]', '', amount_val)
                if not clean_val:
                    return True, None
                value = float(clean_val)
            else:
                value = float(amount_val)
            
            if value <= self.min_amount:
                return False, f"Amount too small in {field_name}: {value} (min {self.min_amount})"
                
            if value > self.max_amount:
                return False, f"Amount too large in {field_name}: {value} (max {self.max_amount})"
                
            return True, None
            
        except ValueError:
            return True, None # Skip if not a number (format check handles this)

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Helper to parse date."""
        from dateutil import parser
        try:
            return parser.parse(date_str)
        except:
            return None
