"""Tests for semantic validation."""

import pytest
from datetime import datetime
from src.validation.semantic import SemanticValidator, ValidationResult


@pytest.fixture
def semantic_config():
    """Sample semantic validation configuration."""
    return {
        'invoice': {
            'required_fields': ['invoice_number', 'date', 'total_amount'],
            'patterns': {
                'invoice_number': '[A-Z0-9-]{6,20}',
                'amount': r'\d+\.?\d{0,2}',
                'date': r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
            },
            'min_amount': 0.01,
            'max_amount': 1000000.0
        },
        'id_document': {
            'required_fields': ['id_number', 'name', 'date_of_birth'],
            'patterns': {
                'id_number': '[A-Z0-9]{8,15}',
                'name': '[A-Z][a-z]+(?: [A-Z][a-z]+)+'
            }
        }
    }


@pytest.fixture
def valid_invoice_fields():
    """Valid invoice fields."""
    return {
        'invoice_number': 'INV-2024-001',
        'date': '2024-01-15',
        'total_amount': 1250.00,
        'vendor_name': 'ACME Corp'
    }


@pytest.fixture
def invalid_invoice_fields():
    """Invalid invoice fields."""
    return {
        'invoice_number': '123',  # Too short
        'date': '2024-13-45',  # Invalid date
        'total_amount': -100.00  # Negative amount
    }


def test_validate_valid_invoice(semantic_config, valid_invoice_fields):
    """Test validation of valid invoice fields."""
    validator = SemanticValidator(semantic_config)
    results = validator.validate_invoice_fields(valid_invoice_fields)
    
    # All fields should be valid
    for field_name, result in results.items():
        if field_name in valid_invoice_fields:
            assert result.is_valid is True


def test_validate_invalid_invoice(semantic_config, invalid_invoice_fields):
    """Test validation of invalid invoice fields."""
    validator = SemanticValidator(semantic_config)
    results = validator.validate_invoice_fields(invalid_invoice_fields)
    
    # Should have validation failures
    assert results['invoice_number'].is_valid is False
    assert results['total_amount'].is_valid is False


def test_missing_required_fields(semantic_config):
    """Test detection of missing required fields."""
    validator = SemanticValidator(semantic_config)
    
    # Missing 'total_amount'
    incomplete_fields = {
        'invoice_number': 'INV-2024-001',
        'date': '2024-01-15'
    }
    
    results = validator.validate_invoice_fields(incomplete_fields)
    
    assert 'total_amount' in results
    assert results['total_amount'].is_valid is False
    assert 'missing' in results['total_amount'].error_message.lower()


def test_pattern_validation(semantic_config):
    """Test regex pattern validation."""
    validator = SemanticValidator(semantic_config)
    
    # Valid pattern
    valid_result = validator._validate_pattern(
        'invoice_number',
        'INV-2024-001',
        '[A-Z0-9-]{6,20}'
    )
    assert valid_result.is_valid is True
    
    # Invalid pattern
    invalid_result = validator._validate_pattern(
        'invoice_number',
        '123',
        '[A-Z0-9-]{6,20}'
    )
    assert invalid_result.is_valid is False


def test_date_validation(semantic_config):
    """Test date validation."""
    validator = SemanticValidator(semantic_config)
    
    # Valid date
    valid_result = validator._validate_date('date', '2024-01-15')
    assert valid_result.is_valid is True
    assert isinstance(valid_result.value, datetime)
    
    # Invalid date
    invalid_result = validator._validate_date('date', 'not-a-date')
    assert invalid_result.is_valid is False


def test_amount_validation(semantic_config):
    """Test amount validation."""
    validator = SemanticValidator(semantic_config)
    
    # Valid amount
    valid_result = validator._validate_amount('total_amount', 100.00, 0.01, 1000000.0)
    assert valid_result.is_valid is True
    
    # Negative amount
    negative_result = validator._validate_amount('total_amount', -50.00, 0.01, 1000000.0)
    assert negative_result.is_valid is False
    
    # Too large amount
    large_result = validator._validate_amount('total_amount', 2000000.00, 0.01, 1000000.0)
    assert large_result.is_valid is False


def test_semantic_score_calculation(semantic_config, valid_invoice_fields):
    """Test semantic score calculation."""
    validator = SemanticValidator(semantic_config)
    
    results = validator.validate_invoice_fields(valid_invoice_fields)
    
    field_weights = {
        'invoice_number': 1.5,
        'date': 1.5,
        'total_amount': 2.0,
        'vendor_name': 1.0
    }
    
    score = validator.calculate_semantic_score(results, field_weights)
    
    assert 0.0 <= score <= 1.0
    # With all valid fields, score should be high
    assert score > 0.8
