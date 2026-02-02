"""Layout validation module (Approach 4).

Validates document structure using spatial anchors and layout rules.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from ..ocr.models import OCRResult, WordData


@dataclass
class AnchorMatch:
    """Represents a matched anchor keyword."""
    keyword: str
    expected_region: str
    actual_position: Tuple[int, int]  # (x, y) center
    matched: bool
    confidence: float


class LayoutValidator:
    """Validates document layout using spatial anchors."""
    
    def __init__(self, config: Dict):
        """Initialize layout validator.
        
        Args:
            config: Layout validation configuration
        """
        self.config = config
    
    def validate_invoice_layout(self, ocr_result: OCRResult, 
                                image_shape: Tuple[int, int]) -> Dict:
        """Validate invoice layout.
        
        Args:
            ocr_result: OCR result with bounding boxes
            image_shape: (height, width) of image
            
        Returns:
            Dictionary with layout validation results
        """
        invoice_config = self.config.get('invoice', {})
        anchors = invoice_config.get('anchors', [])
        min_anchors = invoice_config.get('min_anchors_required', 2)
        
        # Find anchor matches
        anchor_matches = []
        for anchor in anchors:
            match = self._find_anchor(
                ocr_result,
                anchor['keyword'],
                anchor['region'],
                anchor.get('tolerance', 0.3),
                image_shape
            )
            anchor_matches.append(match)
        
        # Calculate score
        matched_count = sum(1 for m in anchor_matches if m.matched)
        total_count = len(anchor_matches)
        
        layout_score = matched_count / total_count if total_count > 0 else 0.0
        passed = matched_count >= min_anchors
        
        return {
            'layout_score': layout_score,
            'passed': passed,
            'anchor_matches': anchor_matches,
            'matched_count': matched_count,
            'total_anchors': total_count
        }
    
    def validate_id_layout(self, ocr_result: OCRResult,
                          image_shape: Tuple[int, int]) -> Dict:
        """Validate ID document layout.
        
        Args:
            ocr_result: OCR result with bounding boxes
            image_shape: (height, width) of image
            
        Returns:
            Dictionary with layout validation results
        """
        id_config = self.config.get('id_document', {})
        anchors = id_config.get('anchors', [])
        min_anchors = id_config.get('min_anchors_required', 2)
        
        # Find anchor matches
        anchor_matches = []
        for anchor in anchors:
            match = self._find_anchor(
                ocr_result,
                anchor['keyword'],
                anchor['region'],
                anchor.get('tolerance', 0.3),
                image_shape
            )
            anchor_matches.append(match)
        
        # Calculate score
        matched_count = sum(1 for m in anchor_matches if m.matched)
        total_count = len(anchor_matches)
        
        layout_score = matched_count / total_count if total_count > 0 else 0.0
        passed = matched_count >= min_anchors
        
        return {
            'layout_score': layout_score,
            'passed': passed,
            'anchor_matches': anchor_matches,
            'matched_count': matched_count,
            'total_anchors': total_count
        }
    
    def _find_anchor(self, ocr_result: OCRResult, keyword: str, 
                    expected_region: str, tolerance: float,
                    image_shape: Tuple[int, int]) -> AnchorMatch:
        """Find anchor keyword in OCR result.
        
        Args:
            ocr_result: OCR result
            keyword: Anchor keyword to find
            expected_region: Expected region (top, bottom, left, right, center)
            tolerance: Position tolerance (fraction of image dimension)
            image_shape: (height, width) of image
            
        Returns:
            AnchorMatch object
        """
        height, width = image_shape
        
        # Search for keyword in OCR words
        best_match = None
        best_confidence = 0.0
        
        for word in ocr_result.words:
            if keyword.lower() in word.text.lower():
                if word.confidence > best_confidence:
                    best_match = word
                    best_confidence = word.confidence
        
        if best_match is None:
            return AnchorMatch(
                keyword=keyword,
                expected_region=expected_region,
                actual_position=(0, 0),
                matched=False,
                confidence=0.0
            )
        
        # Calculate word center position
        x, y, w, h = best_match.bbox
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Check if position matches expected region
        position_matched = self._check_region_match(
            center_x, center_y, width, height, expected_region, tolerance
        )
        
        return AnchorMatch(
            keyword=keyword,
            expected_region=expected_region,
            actual_position=(center_x, center_y),
            matched=position_matched,
            confidence=best_confidence
        )
    
    def _check_region_match(self, x: int, y: int, width: int, height: int,
                           region: str, tolerance: float) -> bool:
        """Check if position matches expected region.
        
        Args:
            x, y: Position coordinates
            width, height: Image dimensions
            region: Expected region name
            tolerance: Position tolerance
            
        Returns:
            True if position matches region
        """
        # Normalize coordinates to [0, 1]
        norm_x = x / width
        norm_y = y / height
        
        if region == 'top':
            return norm_y < tolerance
        elif region == 'bottom':
            return norm_y > (1.0 - tolerance)
        elif region == 'left':
            return norm_x < tolerance
        elif region == 'right':
            return norm_x > (1.0 - tolerance)
        elif region == 'center':
            center_x = 0.5
            center_y = 0.5
            distance = np.sqrt((norm_x - center_x)**2 + (norm_y - center_y)**2)
            return distance < tolerance
        else:
            return True  # Unknown region, accept any position
    
    def detect_photo_region(self, image: np.ndarray, expected_side: str = 'left') -> bool:
        """Detect if photo region exists in expected location (for ID documents).
        
        Args:
            image: Input image
            expected_side: Expected side of photo (left, right)
            
        Returns:
            True if photo region detected in expected location
        """
        import cv2
        
        height, width = image.shape[:2]
        
        # Define region to check
        if expected_side == 'left':
            region = image[:, :width//3]
        elif expected_side == 'right':
            region = image[:, 2*width//3:]
        else:
            return False
        
        # Convert to grayscale
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region
        
        # Detect faces using Haar Cascade (simple approach)
        try:
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            return len(faces) > 0
        except:
            # If face detection fails, use edge density as proxy
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.count_nonzero(edges) / edges.size
            
            # Photo regions typically have moderate edge density
            return 0.05 < edge_density < 0.3
