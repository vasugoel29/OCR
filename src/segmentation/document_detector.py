"""Contour-based document detection for identifying document regions."""

import logging
from typing import List, Tuple
import cv2
import numpy as np

from .region import Region, BoundingBox


class DocumentDetector:
    """Detect document regions using contour analysis."""
    
    def __init__(self, config: dict = None):
        """Initialize document detector.
        
        Args:
            config: Configuration dictionary with detection parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Get configuration parameters
        contour_config = self.config.get('contour_detection', {})
        self.min_area_ratio = contour_config.get('min_area_ratio', 0.20)
        self.max_area_ratio = contour_config.get('max_area_ratio', 0.95)
        self.min_aspect_ratio = contour_config.get('min_aspect_ratio', 0.3)
        self.max_aspect_ratio = contour_config.get('max_aspect_ratio', 3.0)
        self.approx_epsilon = contour_config.get('approx_epsilon', 0.02)
    
    def detect_contours(self, image: np.ndarray) -> List[Region]:
        """Detect document regions using contour detection.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            List of detected regions
        """
        self.logger.debug("Starting contour-based document detection")
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate edges to close gaps
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        self.logger.debug(f"Found {len(contours)} contours")
        
        # Filter and extract regions
        regions = []
        image_area = image.shape[0] * image.shape[1]
        
        for contour in contours:
            if self._is_valid_document_contour(contour, image.shape, image_area):
                region = self._extract_region_from_contour(image, contour, image_area)
                if region:
                    regions.append(region)
        
        self.logger.info(f"Detected {len(regions)} valid document regions using contours")
        return regions
    
    def _is_valid_document_contour(self, contour: np.ndarray, 
                                   image_shape: Tuple, 
                                   image_area: int) -> bool:
        """Check if contour represents a valid document region.
        
        Args:
            contour: Contour to validate
            image_shape: Shape of the image (height, width, ...)
            image_area: Total area of the image
            
        Returns:
            True if contour is valid document region
        """
        # Calculate contour area
        area = cv2.contourArea(contour)
        area_ratio = area / image_area
        
        # Check area ratio
        if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
            return False
        
        # Approximate contour to polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, self.approx_epsilon * peri, True)
        
        # Check if approximately quadrilateral (4-6 vertices)
        # Allow some flexibility for imperfect rectangles
        if not (4 <= len(approx) <= 6):
            return False
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Check aspect ratio
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
            return False
        
        # Check if bounding box is reasonable (not at extreme edges)
        margin = 5  # pixels
        if x < margin or y < margin:
            return False
        if x + w > image_shape[1] - margin or y + h > image_shape[0] - margin:
            # Allow if it's a large region (likely the whole document)
            if area_ratio < 0.8:
                return False
        
        return True
    
    def _extract_region_from_contour(self, image: np.ndarray, 
                                     contour: np.ndarray,
                                     image_area: int) -> Region:
        """Extract region from a valid contour.
        
        Args:
            image: Source image
            contour: Valid document contour
            image_area: Total image area
            
        Returns:
            Region object or None if extraction fails
        """
        try:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Extract region image
            region_image = image[y:y+h, x:x+w].copy()
            
            # Calculate confidence based on contour properties
            area = cv2.contourArea(contour)
            area_ratio = area / image_area
            
            # Higher confidence for larger, more rectangular regions
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, self.approx_epsilon * peri, True)
            rectangularity = len(approx) / 4.0  # Closer to 1.0 is better
            
            # Confidence: weighted by area and shape
            confidence = min(1.0, area_ratio * 0.7 + (1.0 / rectangularity) * 0.3)
            
            return Region(
                bbox=BoundingBox(x, y, w, h),
                image=region_image,
                confidence=confidence,
                detection_method='contour',
                area_ratio=area_ratio,
                contour=contour
            )
        
        except Exception as e:
            self.logger.warning(f"Failed to extract region from contour: {e}")
            return None
