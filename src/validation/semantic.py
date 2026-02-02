"""Semantic validation module (Approach 3).

Validates extracted fields using regex and format rules.
"""

import re
from typing import Dict, List, Optional
from datetime import datetime
from dateutil import parser as date_parser
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of field validation."""
    is_valid: bool
    field_name: str
    value: any
    error_message: Optional[str] = None


class SemanticValidator:
    """Validates document fields using semantic rules."""
    
    def __init__(self, config: Dict):
        """Initialize semantic validator.
        
        Args:
            config: Semantic validation configuration
        """
        self.config = config
    
    def validate_invoice_fields(self, fields: Dict[str, any]) -> Dict[str, ValidationResult]:
        """Validate invoice fields.
        
        Args:
            fields: Extracted invoice fields
            
        Returns:
            Dictionary of field names to validation results
        """
        invoice_config = self.config.get('invoice', {})
        required_fields = invoice_config.get('required_fields', [])
        patterns = invoice_config.get('patterns', {})
        
        results = {}
        
        # Check required fields
        for field in required_fields:
            if field not in fields:
                results[field] = ValidationResult(
                    is_valid=False,
                    field_name=field,
                    value=None,
                    error_message=f"Required field '{field}' is missing"
                )
        
        # Validate invoice number
        if 'invoice_number' in fields:
            results['invoice_number'] = self._validate_pattern(
                'invoice_number',
                fields['invoice_number'],
                patterns.get('invoice_number')
            )
        
        # Validate date
        if 'date' in fields:
            results['date'] = self._validate_date(
                'date',
                fields['date']
            )
        
        # Validate amounts
        if 'total_amount' in fields:
            results['total_amount'] = self._validate_amount(
                'total_amount',
                fields['total_amount'],
                invoice_config.get('min_amount', 0.01),
                invoice_config.get('max_amount', 1000000.0)
            )
        
        if 'subtotal' in fields:
            results['subtotal'] = self._validate_amount(
                'subtotal',
                fields['subtotal'],
                invoice_config.get('min_amount', 0.01),
                invoice_config.get('max_amount', 1000000.0)
            )
        
        if 'tax' in fields:
            results['tax'] = self._validate_amount(
                'tax',
                fields['tax'],
                0.0,
                invoice_config.get('max_amount', 1000000.0)
            )
        
        return results
    
    def validate_id_fields(self, fields: Dict[str, any]) -> Dict[str, ValidationResult]:
        """Validate ID document fields.
        
        Args:
            fields: Extracted ID fields
            
        Returns:
            Dictionary of field names to validation results
        """
        id_config = self.config.get('id_document', {})
        required_fields = id_config.get('required_fields', [])
        patterns = id_config.get('patterns', {})
        
        results = {}
        
        # Check required fields
        for field in required_fields:
            if field not in fields:
                results[field] = ValidationResult(
                    is_valid=False,
                    field_name=field,
                    value=None,
                    error_message=f"Required field '{field}' is missing"
                )
        
        # Validate ID number
        if 'id_number' in fields:
            results['id_number'] = self._validate_pattern(
                'id_number',
                fields['id_number'],
                patterns.get('id_number')
            )
        
        # Validate name
        if 'name' in fields:
            results['name'] = self._validate_pattern(
                'name',
                fields['name'],
                patterns.get('name')
            )
        
        # Validate dates
        if 'date_of_birth' in fields:
            results['date_of_birth'] = self._validate_date(
                'date_of_birth',
                fields['date_of_birth']
            )
        
        if 'issue_date' in fields:
            results['issue_date'] = self._validate_date(
                'issue_date',
                fields['issue_date']
            )
        
        if 'expiry_date' in fields:
            results['expiry_date'] = self._validate_date(
                'expiry_date',
                fields['expiry_date']
            )
        
        return results
    
    def calculate_semantic_score(self, validation_results: Dict[str, ValidationResult],
                                  field_weights: Dict[str, float]) -> float:
        """Calculate semantic validation score.
        
        Args:
            validation_results: Dictionary of validation results
            field_weights: Importance weights for each field
            
        Returns:
            Semantic score normalized to [0, 1]
        """
        if not validation_results:
            return 0.0
        
        total_weight = 0.0
        weighted_valid = 0.0
        
        for field_name, result in validation_results.items():
            weight = field_weights.get(field_name, 1.0)
            total_weight += weight
            
            if result.is_valid:
                weighted_valid += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_valid / total_weight
    
    def _validate_pattern(self, field_name: str, value: str, pattern: Optional[str]) -> ValidationResult:
        """Validate field against regex pattern.
        
        Args:
            field_name: Name of field
            value: Field value
            pattern: Regex pattern
            
        Returns:
            ValidationResult
        """
        if pattern is None:
            return ValidationResult(
                is_valid=True,
                field_name=field_name,
                value=value
            )
        
        if re.match(pattern, str(value)):
            return ValidationResult(
                is_valid=True,
                field_name=field_name,
                value=value
            )
        else:
            return ValidationResult(
                is_valid=False,
                field_name=field_name,
                value=value,
                error_message=f"Value '{value}' does not match pattern '{pattern}'"
            )
    
    def _validate_date(self, field_name: str, value: str) -> ValidationResult:
        """Validate date field.
        
        Args:
            field_name: Name of field
            value: Date string
            
        Returns:
            ValidationResult
        """
        try:
            # Try to parse date
            parsed_date = date_parser.parse(value, fuzzy=True)
            
            # Check if date is reasonable (not too far in past/future)
            current_year = datetime.now().year
            if parsed_date.year < 1900 or parsed_date.year > current_year + 100:
                return ValidationResult(
                    is_valid=False,
                    field_name=field_name,
                    value=value,
                    error_message=f"Date year {parsed_date.year} is out of reasonable range"
                )
            
            return ValidationResult(
                is_valid=True,
                field_name=field_name,
                value=parsed_date
            )
        except (ValueError, TypeError) as e:
            return ValidationResult(
                is_valid=False,
                field_name=field_name,
                value=value,
                error_message=f"Invalid date format: {str(e)}"
            )
    
    def _validate_amount(self, field_name: str, value: float, 
                        min_amount: float, max_amount: float) -> ValidationResult:
        """Validate monetary amount.
        
        Args:
            field_name: Name of field
            value: Amount value
            min_amount: Minimum valid amount
            max_amount: Maximum valid amount
            
        Returns:
            ValidationResult
        """
        try:
            amount = float(value)
            
            if amount < min_amount:
                return ValidationResult(
                    is_valid=False,
                    field_name=field_name,
                    value=value,
                    error_message=f"Amount {amount} is below minimum {min_amount}"
                )
            
            if amount > max_amount:
                return ValidationResult(
                    is_valid=False,
                    field_name=field_name,
                    value=value,
                    error_message=f"Amount {amount} exceeds maximum {max_amount}"
                )
            
            return ValidationResult(
                is_valid=True,
                field_name=field_name,
                value=amount
            )
        except (ValueError, TypeError) as e:
            return ValidationResult(
                is_valid=False,
                field_name=field_name,
                value=value,
                error_message=f"Invalid amount format: {str(e)}"
            )
