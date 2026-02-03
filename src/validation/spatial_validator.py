"""Spatial validation to prevent cross-region field mixing."""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np


class SpatialValidator:
    """Validate spatial compactness of extracted fields to prevent cross-region mixing."""
    
    def __init__(self, config: dict = None):
        """Initialize spatial validator.
        
        Args:
            config: Configuration dictionary with spatial validation parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Get configuration parameters
        spatial_config = self.config.get('spatial_validation', {})
        self.enabled = spatial_config.get('enabled', True)
        self.max_field_dispersion = spatial_config.get('max_field_dispersion', 0.5)
        self.require_single_schema = spatial_config.get('require_single_schema', True)
    
    def validate_field_compactness(self, 
                                   extracted_fields: Dict,
                                   ocr_boxes: List[Tuple],
                                   ocr_texts: List[str]) -> Tuple[float, Dict]:
        """Validate that extracted fields are spatially compact.
        
        Args:
            extracted_fields: Dictionary of extracted field values
            ocr_boxes: List of bounding boxes from OCR (x, y, w, h)
            ocr_texts: List of text strings corresponding to boxes
            
        Returns:
            Tuple of (spatial_score, details_dict)
        """
        if not self.enabled:
            return 1.0, {'enabled': False}
        
        if not extracted_fields or not ocr_boxes:
            return 1.0, {'no_fields': True}
        
        # Map field values to their bounding boxes
        field_boxes = self._map_fields_to_boxes(extracted_fields, ocr_boxes, ocr_texts)
        
        if not field_boxes:
            self.logger.debug("No field boxes found for spatial validation")
            return 0.5, {'no_field_boxes': True}
        
        # Calculate spatial dispersion
        dispersion = self._calculate_dispersion(field_boxes)
        
        # Calculate spatial score (lower dispersion = higher score)
        # Normalize dispersion to 0-1 range
        normalized_dispersion = min(1.0, dispersion / self.max_field_dispersion)
        spatial_score = 1.0 - normalized_dispersion
        
        # Detect multiple schema clusters
        num_clusters = self._detect_schema_clusters(field_boxes)
        
        details = {
            'dispersion': dispersion,
            'normalized_dispersion': normalized_dispersion,
            'num_clusters': num_clusters,
            'field_count': len(field_boxes),
            'spatial_score': spatial_score
        }
        
        # Penalize if multiple schemas detected
        if self.require_single_schema and num_clusters > 1:
            self.logger.warning(f"Multiple schema clusters detected: {num_clusters}")
            spatial_score *= 0.5  # Heavy penalty
            details['multiple_schemas_penalty'] = True
        
        self.logger.debug(f"Spatial validation score: {spatial_score:.3f} (dispersion: {dispersion:.3f})")
        
        return spatial_score, details
    
    def _map_fields_to_boxes(self, 
                            extracted_fields: Dict,
                            ocr_boxes: List[Tuple],
                            ocr_texts: List[str]) -> List[Tuple]:
        """Map extracted field values to their OCR bounding boxes.
        
        Args:
            extracted_fields: Dictionary of field values
            ocr_boxes: List of OCR bounding boxes
            ocr_texts: List of OCR text strings
            
        Returns:
            List of (field_name, bbox) tuples
        """
        field_boxes = []
        
        for field_name, field_value in extracted_fields.items():
            if not field_value or not isinstance(field_value, str):
                continue
            
            # Clean field value for matching
            field_value_clean = str(field_value).strip().lower()
            
            # Find matching OCR text
            for idx, ocr_text in enumerate(ocr_texts):
                ocr_text_clean = ocr_text.strip().lower()
                
                # Check if field value is in OCR text or vice versa
                if field_value_clean in ocr_text_clean or ocr_text_clean in field_value_clean:
                    if idx < len(ocr_boxes):
                        field_boxes.append((field_name, ocr_boxes[idx]))
                        break
        
        return field_boxes
    
    def _calculate_dispersion(self, field_boxes: List[Tuple]) -> float:
        """Calculate spatial dispersion of field bounding boxes.
        
        Args:
            field_boxes: List of (field_name, bbox) tuples
            
        Returns:
            Dispersion score (0 = compact, higher = more dispersed)
        """
        if len(field_boxes) < 2:
            return 0.0
        
        # Extract centers of bounding boxes
        centers = []
        for _, bbox in field_boxes:
            if len(bbox) >= 4:
                x, y, w, h = bbox[:4]
                centers.append([x + w/2, y + h/2])
        
        if len(centers) < 2:
            return 0.0
        
        centers = np.array(centers)
        
        # Calculate standard deviation of positions
        std_x = np.std(centers[:, 0])
        std_y = np.std(centers[:, 1])
        
        # Combined dispersion (normalized by image size would be better, but we don't have it here)
        dispersion = np.sqrt(std_x**2 + std_y**2) / 1000.0  # Rough normalization
        
        return dispersion
    
    def _detect_schema_clusters(self, field_boxes: List[Tuple]) -> int:
        """Detect number of distinct spatial clusters of fields.
        
        Args:
            field_boxes: List of (field_name, bbox) tuples
            
        Returns:
            Number of detected clusters
        """
        if len(field_boxes) < 2:
            return 1
        
        # Extract centers
        centers = []
        for _, bbox in field_boxes:
            if len(bbox) >= 4:
                x, y, w, h = bbox[:4]
                centers.append([x + w/2, y + h/2])
        
        if len(centers) < 2:
            return 1
        
        centers = np.array(centers)
        
        # Simple clustering: if any two points are very far apart, consider multiple clusters
        # Calculate pairwise distances
        max_distance = 0
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                dist = np.linalg.norm(centers[i] - centers[j])
                max_distance = max(max_distance, dist)
        
        # If max distance is very large, likely multiple documents
        # Threshold: if max distance > 500 pixels, consider multiple clusters
        if max_distance > 500:
            return 2
        
        return 1
    
    def calculate_spatial_score(self, 
                               extracted_fields: Dict,
                               ocr_boxes: List[Tuple],
                               ocr_texts: List[str]) -> float:
        """Calculate spatial compactness score (convenience method).
        
        Args:
            extracted_fields: Dictionary of field values
            ocr_boxes: List of OCR bounding boxes
            ocr_texts: List of OCR text strings
            
        Returns:
            Spatial score (0-1)
        """
        score, _ = self.validate_field_compactness(extracted_fields, ocr_boxes, ocr_texts)
        return score
