"""Test multi-document detection and segmentation."""

import pytest
import numpy as np
import cv2
from src.segmentation import SegmentationPipeline, Region, BoundingBox


def test_single_document_detection():
    """Test detection of single document (full image)."""
    # Create simple test image
    image = np.ones((1000, 800, 3), dtype=np.uint8) * 255
    
    # Initialize segmentation pipeline
    config = {'enabled': True}
    pipeline = SegmentationPipeline(config)
    
    # Detect regions
    regions = pipeline.detect_regions(image)
    
    # Should detect 1 region (full image)
    assert len(regions) == 1
    assert regions[0].detection_method == 'full_image'
    assert regions[0].area_ratio == 1.0


def test_region_bounding_box():
    """Test bounding box utilities."""
    bbox1 = BoundingBox(10, 10, 100, 100)
    bbox2 = BoundingBox(50, 50, 100, 100)
    bbox3 = BoundingBox(200, 200, 100, 100)
    
    # Test area calculation
    assert bbox1.area == 10000
    
    # Test center calculation
    assert bbox1.center == (60, 60)
    
    # Test aspect ratio
    assert bbox1.aspect_ratio == 1.0
    
    # Test overlap detection
    assert bbox1.overlaps_with(bbox2, threshold=0.1)  # Should overlap
    assert not bbox1.overlaps_with(bbox3, threshold=0.1)  # Should not overlap


def test_region_validation():
    """Test region data validation."""
    image = np.ones((100, 100, 3), dtype=np.uint8)
    bbox = BoundingBox(0, 0, 100, 100)
    
    # Valid region
    region = Region(
        bbox=bbox,
        image=image,
        confidence=0.8,
        detection_method='contour',
        area_ratio=0.5
    )
    assert region.confidence == 0.8
    
    # Invalid confidence
    with pytest.raises(ValueError):
        Region(
            bbox=bbox,
            image=image,
            confidence=1.5,  # Invalid
            detection_method='contour',
            area_ratio=0.5
        )
    
    # Invalid detection method
    with pytest.raises(ValueError):
        Region(
            bbox=bbox,
            image=image,
            confidence=0.8,
            detection_method='invalid',  # Invalid
            area_ratio=0.5
        )


def test_segmentation_disabled():
    """Test that segmentation can be disabled."""
    image = np.ones((1000, 800, 3), dtype=np.uint8) * 255
    
    # Disable segmentation
    config = {'enabled': False}
    pipeline = SegmentationPipeline(config)
    
    # Should return full image
    regions = pipeline.detect_regions(image)
    assert len(regions) == 1
    assert regions[0].detection_method == 'full_image'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
