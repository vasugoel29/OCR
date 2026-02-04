"""Tests for validation modules."""

import pytest
from datetime import datetime
from ocr_pipeline.validation.business_rules import BusinessRuleValidator


@pytest.fixture
def validation_config():
    """Sample validation configuration."""
    return {
        'date_validation': {
            'reject_future_dates': True,
            'max_age_days': 365
        },
        'amount_validation': {
            'min_amount': 0.01,
            'max_amount': 1000000.0
        }
    }


@pytest.fixture
def valid_fields():
    """Valid document fields."""
    return {
        'invoice_number': 'INV-2024-001',
        'date': '2024-01-15',
        'total_amount': 1250.00,
        'vendor_name': 'ACME Corp'
    }


@pytest.fixture
def invalid_fields():
    """Invalid document fields."""
    return {
        'invoice_number': '123',  # Too short
        'date': '2025-13-45',  # Invalid/future date
        'total_amount': -100.00  # Negative amount
    }


def test_business_rule_validator_init(validation_config):
    """Test BusinessRuleValidator initialization."""
    validator = BusinessRuleValidator(validation_config)
    assert validator.config == validation_config
    assert validator.reject_future_dates is True
    assert validator.max_age_days == 365


def test_validate_valid_fields(validation_config, valid_fields):
    """Test validation of valid fields."""
    validator = BusinessRuleValidator(validation_config)
    is_valid, reasons = validator.validate(valid_fields, 'invoice')
    
    # Should pass validation
    assert is_valid is True
    assert len(reasons) == 0


def test_validate_invalid_fields(validation_config, invalid_fields):
    """Test validation of invalid fields."""
    validator = BusinessRuleValidator(validation_config)
    is_valid, reasons = validator.validate(invalid_fields, 'invoice')
    
    # Should fail validation
    assert is_valid is False
    assert len(reasons) > 0


def test_date_validation(validation_config):
    """Test date validation logic."""
    validator = BusinessRuleValidator(validation_config)
    
    # Valid past date
    valid_date = '2023-01-15'
    is_valid, _ = validator._validate_date('date', valid_date)
    assert is_valid is True
    
    # Future date (if reject_future_dates is True)
    if validator.reject_future_dates:
        future_date = '2025-12-31'
        is_valid, _ = validator._validate_date('date', future_date)
        assert is_valid is False


def test_amount_validation(validation_config):
    """Test amount validation logic."""
    validator = BusinessRuleValidator(validation_config)
    
    # Valid amount
    valid_amount = 100.00
    is_valid, _ = validator._validate_amount('total_amount', valid_amount)
    assert is_valid is True
    
    # Negative amount
    negative_amount = -50.00
    is_valid, _ = validator._validate_amount('total_amount', negative_amount)
    assert is_valid is False
    
    # Too large amount
    large_amount = 2000000.00
    is_valid, _ = validator._validate_amount('total_amount', large_amount)
    assert is_valid is False
