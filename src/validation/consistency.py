"""Statistical consistency validation module (Approach 5).

Validates cross-field logical relationships and arithmetic consistency.
"""

from typing import Dict, Optional
from datetime import datetime, timedelta
from dateutil import parser as date_parser
from dataclasses import dataclass


@dataclass
class ConsistencyCheck:
    """Result of a consistency check."""
    check_name: str
    passed: bool
    error_message: Optional[str] = None


class ConsistencyValidator:
    """Validates statistical and logical consistency of extracted fields."""
    
    def __init__(self, config: Dict):
        """Initialize consistency validator.
        
        Args:
            config: Consistency validation configuration
        """
        self.config = config
    
    def validate_invoice_consistency(self, fields: Dict[str, any]) -> Dict[str, ConsistencyCheck]:
        """Validate invoice field consistency.
        
        Args:
            fields: Extracted invoice fields
            
        Returns:
            Dictionary of consistency check results
        """
        invoice_config = self.config.get('invoice', {})
        checks = {}
        
        # Arithmetic consistency check
        if invoice_config.get('enable_arithmetic_check', True):
            arithmetic_check = self._check_invoice_arithmetic(
                fields,
                invoice_config.get('arithmetic_tolerance', 0.02)
            )
            checks['arithmetic'] = arithmetic_check
        
        # Date validation
        if invoice_config.get('enable_date_validation', True):
            date_check = self._check_invoice_dates(
                fields,
                invoice_config.get('max_future_days', 30)
            )
            checks['date_validity'] = date_check
        
        return checks
    
    def validate_id_consistency(self, fields: Dict[str, any]) -> Dict[str, ConsistencyCheck]:
        """Validate ID document field consistency.
        
        Args:
            fields: Extracted ID fields
            
        Returns:
            Dictionary of consistency check results
        """
        id_config = self.config.get('id_document', {})
        checks = {}
        
        # Date validation
        if id_config.get('enable_date_validation', True):
            date_check = self._check_id_dates(fields)
            checks['date_validity'] = date_check
        
        # Expiry after issue check
        if id_config.get('require_expiry_after_issue', True):
            expiry_check = self._check_expiry_after_issue(fields)
            if expiry_check:
                checks['expiry_after_issue'] = expiry_check
        
        # Age validation
        age_check = self._check_age_validity(
            fields,
            id_config.get('min_age', 0),
            id_config.get('max_age', 150)
        )
        if age_check:
            checks['age_validity'] = age_check
        
        return checks
    
    def calculate_consistency_score(self, checks: Dict[str, ConsistencyCheck]) -> float:
        """Calculate consistency score.
        
        Args:
            checks: Dictionary of consistency checks
            
        Returns:
            Consistency score normalized to [0, 1]
        """
        if not checks:
            return 1.0  # No checks to fail
        
        passed_count = sum(1 for check in checks.values() if check.passed)
        total_count = len(checks)
        
        return passed_count / total_count
    
    def _check_invoice_arithmetic(self, fields: Dict[str, any], 
                                  tolerance: float) -> ConsistencyCheck:
        """Check if subtotal + tax = total.
        
        Args:
            fields: Invoice fields
            tolerance: Allowed tolerance (fraction)
            
        Returns:
            ConsistencyCheck result
        """
        if 'total_amount' not in fields:
            return ConsistencyCheck(
                check_name='arithmetic',
                passed=True,  # Can't check without total
                error_message=None
            )
        
        total = float(fields['total_amount'])
        
        # If we have subtotal and tax, check arithmetic
        if 'subtotal' in fields and 'tax' in fields:
            subtotal = float(fields['subtotal'])
            tax = float(fields['tax'])
            calculated_total = subtotal + tax
            
            # Check if within tolerance
            difference = abs(calculated_total - total)
            allowed_difference = total * tolerance
            
            if difference <= allowed_difference:
                return ConsistencyCheck(
                    check_name='arithmetic',
                    passed=True
                )
            else:
                return ConsistencyCheck(
                    check_name='arithmetic',
                    passed=False,
                    error_message=f"Arithmetic mismatch: {subtotal} + {tax} = {calculated_total}, but total is {total}"
                )
        
        # If we only have subtotal, it should be <= total
        if 'subtotal' in fields:
            subtotal = float(fields['subtotal'])
            if subtotal > total:
                return ConsistencyCheck(
                    check_name='arithmetic',
                    passed=False,
                    error_message=f"Subtotal {subtotal} exceeds total {total}"
                )
        
        return ConsistencyCheck(
            check_name='arithmetic',
            passed=True
        )
    
    def _check_invoice_dates(self, fields: Dict[str, any],
                            max_future_days: int) -> ConsistencyCheck:
        """Check if invoice dates are valid.
        
        Args:
            fields: Invoice fields
            max_future_days: Maximum days in future allowed
            
        Returns:
            ConsistencyCheck result
        """
        if 'date' not in fields:
            return ConsistencyCheck(
                check_name='date_validity',
                passed=True
            )
        
        try:
            # Parse invoice date
            if isinstance(fields['date'], datetime):
                invoice_date = fields['date']
            else:
                invoice_date = date_parser.parse(str(fields['date']))
            
            # Check if date is not too far in future
            max_future_date = datetime.now() + timedelta(days=max_future_days)
            if invoice_date > max_future_date:
                return ConsistencyCheck(
                    check_name='date_validity',
                    passed=False,
                    error_message=f"Invoice date {invoice_date} is too far in future"
                )
            
            # If due date exists, it should be after invoice date
            if 'due_date' in fields:
                if isinstance(fields['due_date'], datetime):
                    due_date = fields['due_date']
                else:
                    due_date = date_parser.parse(str(fields['due_date']))
                
                if due_date < invoice_date:
                    return ConsistencyCheck(
                        check_name='date_validity',
                        passed=False,
                        error_message=f"Due date {due_date} is before invoice date {invoice_date}"
                    )
            
            return ConsistencyCheck(
                check_name='date_validity',
                passed=True
            )
        except (ValueError, TypeError) as e:
            return ConsistencyCheck(
                check_name='date_validity',
                passed=False,
                error_message=f"Date parsing error: {str(e)}"
            )
    
    def _check_id_dates(self, fields: Dict[str, any]) -> ConsistencyCheck:
        """Check if ID dates are valid.
        
        Args:
            fields: ID fields
            
        Returns:
            ConsistencyCheck result
        """
        try:
            # Check if DOB is in the past
            if 'date_of_birth' in fields:
                if isinstance(fields['date_of_birth'], datetime):
                    dob = fields['date_of_birth']
                else:
                    dob = date_parser.parse(str(fields['date_of_birth']))
                
                if dob > datetime.now():
                    return ConsistencyCheck(
                        check_name='date_validity',
                        passed=False,
                        error_message="Date of birth is in the future"
                    )
            
            # Check if expiry is in the future (or recently past)
            if 'expiry_date' in fields:
                if isinstance(fields['expiry_date'], datetime):
                    expiry = fields['expiry_date']
                else:
                    expiry = date_parser.parse(str(fields['expiry_date']))
                
                # Allow up to 1 year expired
                min_expiry = datetime.now() - timedelta(days=365)
                if expiry < min_expiry:
                    return ConsistencyCheck(
                        check_name='date_validity',
                        passed=False,
                        error_message=f"ID expired too long ago: {expiry}"
                    )
            
            return ConsistencyCheck(
                check_name='date_validity',
                passed=True
            )
        except (ValueError, TypeError) as e:
            return ConsistencyCheck(
                check_name='date_validity',
                passed=False,
                error_message=f"Date parsing error: {str(e)}"
            )
    
    def _check_expiry_after_issue(self, fields: Dict[str, any]) -> Optional[ConsistencyCheck]:
        """Check if expiry date is after issue date.
        
        Args:
            fields: ID fields
            
        Returns:
            ConsistencyCheck result or None
        """
        if 'issue_date' not in fields or 'expiry_date' not in fields:
            return None
        
        try:
            if isinstance(fields['issue_date'], datetime):
                issue_date = fields['issue_date']
            else:
                issue_date = date_parser.parse(str(fields['issue_date']))
            
            if isinstance(fields['expiry_date'], datetime):
                expiry_date = fields['expiry_date']
            else:
                expiry_date = date_parser.parse(str(fields['expiry_date']))
            
            if expiry_date <= issue_date:
                return ConsistencyCheck(
                    check_name='expiry_after_issue',
                    passed=False,
                    error_message=f"Expiry date {expiry_date} is not after issue date {issue_date}"
                )
            
            return ConsistencyCheck(
                check_name='expiry_after_issue',
                passed=True
            )
        except (ValueError, TypeError) as e:
            return ConsistencyCheck(
                check_name='expiry_after_issue',
                passed=False,
                error_message=f"Date parsing error: {str(e)}"
            )
    
    def _check_age_validity(self, fields: Dict[str, any],
                           min_age: int, max_age: int) -> Optional[ConsistencyCheck]:
        """Check if age is within valid range.
        
        Args:
            fields: ID fields
            min_age: Minimum valid age
            max_age: Maximum valid age
            
        Returns:
            ConsistencyCheck result or None
        """
        if 'date_of_birth' not in fields:
            return None
        
        try:
            if isinstance(fields['date_of_birth'], datetime):
                dob = fields['date_of_birth']
            else:
                dob = date_parser.parse(str(fields['date_of_birth']))
            
            # Calculate age
            today = datetime.now()
            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            
            if age < min_age or age > max_age:
                return ConsistencyCheck(
                    check_name='age_validity',
                    passed=False,
                    error_message=f"Age {age} is outside valid range [{min_age}, {max_age}]"
                )
            
            return ConsistencyCheck(
                check_name='age_validity',
                passed=True
            )
        except (ValueError, TypeError) as e:
            return ConsistencyCheck(
                check_name='age_validity',
                passed=False,
                error_message=f"Age calculation error: {str(e)}"
            )
