"""ID document processor."""

from typing import Dict, Any
import numpy as np
from .base import BaseDocumentProcessor
from ..ocr.models import OCRResult
from ..validation.extractors import FieldExtractor
from ..validation.semantic import SemanticValidator
from ..validation.layout import LayoutValidator
from ..validation.consistency import ConsistencyValidator


class IDDocumentProcessor(BaseDocumentProcessor):
    """Processes ID documents."""
    
    def __init__(self, config: Dict):
        """Initialize ID document processor.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        self.extractor = FieldExtractor(config.get('semantic', {}))
        self.semantic_validator = SemanticValidator(config.get('semantic', {}))
        self.layout_validator = LayoutValidator(config.get('layout', {}))
        self.consistency_validator = ConsistencyValidator(config.get('consistency', {}))
    
    def extract_fields(self, ocr_result: OCRResult) -> Dict[str, Any]:
        """Extract ID fields from OCR result.
        
        Args:
            ocr_result: OCR result object
            
        Returns:
            Dictionary of extracted ID fields
        """
        return self.extractor.extract_id_fields(ocr_result)
    
    def validate_fields(self, fields: Dict[str, Any]) -> Dict:
        """Validate ID fields.
        
        Args:
            fields: Extracted ID fields
            
        Returns:
            Validation results with score
        """
        validation_results = self.semantic_validator.validate_id_fields(fields)
        
        # Get field weights
        field_weights = self.config.get('scoring', {}).get('field_weights', {}).get('id_document', {})
        
        # Calculate semantic score
        semantic_score = self.semantic_validator.calculate_semantic_score(
            validation_results,
            field_weights
        )
        
        return {
            'validation_results': validation_results,
            'semantic_score': semantic_score
        }
    
    def validate_layout(self, ocr_result: OCRResult, image_shape: tuple) -> Dict:
        """Validate ID document layout.
        
        Args:
            ocr_result: OCR result
            image_shape: (height, width) of image
            
        Returns:
            Layout validation results
        """
        return self.layout_validator.validate_id_layout(ocr_result, image_shape)
    
    def check_consistency(self, fields: Dict[str, Any]) -> Dict:
        """Check ID field consistency.
        
        Args:
            fields: Extracted ID fields
            
        Returns:
            Consistency check results with score
        """
        consistency_checks = self.consistency_validator.validate_id_consistency(fields)
        
        # Calculate consistency score
        consistency_score = self.consistency_validator.calculate_consistency_score(
            consistency_checks
        )
        
        return {
            'consistency_checks': consistency_checks,
            'consistency_score': consistency_score
        }
    
    def get_document_type(self) -> str:
        """Get document type identifier.
        
        Returns:
            'id_document'
        """
        return 'id_document'
    
    def get_required_fields(self) -> list:
        """Get list of required fields for ID documents.
        
        Returns:
            List of required field names
        """
        return self.config.get('semantic', {}).get('id_document', {}).get('required_fields', [])
