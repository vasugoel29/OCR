"""Data structures for document regions and bounding boxes."""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class BoundingBox:
    """Bounding box for a region in an image."""
    x: int
    y: int
    width: int
    height: int
    
    @property
    def area(self) -> int:
        """Calculate area of bounding box."""
        return self.width * self.height
    
    @property
    def center(self) -> tuple:
        """Calculate center point of bounding box."""
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def aspect_ratio(self) -> float:
        """Calculate aspect ratio (width/height)."""
        if self.height == 0:
            return 0.0
        return self.width / self.height
    
    def to_tuple(self) -> tuple:
        """Convert to (x, y, width, height) tuple."""
        return (self.x, self.y, self.width, self.height)
    
    def to_corners(self) -> tuple:
        """Convert to (x1, y1, x2, y2) corner coordinates."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)
    
    def overlaps_with(self, other: 'BoundingBox', threshold: float = 0.5) -> bool:
        """Check if this box overlaps with another box.
        
        Args:
            other: Another bounding box
            threshold: Minimum IoU to consider as overlap
            
        Returns:
            True if boxes overlap above threshold
        """
        # Calculate intersection
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.width, other.x + other.width)
        y2 = min(self.y + self.height, other.y + other.height)
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        intersection = (x2 - x1) * (y2 - y1)
        union = self.area + other.area - intersection
        
        iou = intersection / union if union > 0 else 0
        return iou >= threshold


@dataclass
class Region:
    """Detected document region in an image."""
    bbox: BoundingBox
    image: np.ndarray
    confidence: float
    detection_method: str  # 'contour', 'text_cluster', or 'merged'
    area_ratio: float  # Ratio to total image area
    contour: Optional[np.ndarray] = None  # Original contour if from contour detection
    
    def __post_init__(self):
        """Validate region data."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
        
        if not 0.0 <= self.area_ratio <= 1.0:
            raise ValueError(f"Area ratio must be between 0 and 1, got {self.area_ratio}")
        
        if self.detection_method not in ['contour', 'text_cluster', 'merged', 'full_image']:
            raise ValueError(f"Invalid detection method: {self.detection_method}")
    
    def to_dict(self) -> dict:
        """Convert region to dictionary for serialization."""
        return {
            'bbox': self.bbox.to_tuple(),
            'confidence': self.confidence,
            'detection_method': self.detection_method,
            'area_ratio': self.area_ratio,
            'aspect_ratio': self.bbox.aspect_ratio
        }
