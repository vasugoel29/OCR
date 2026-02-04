"""Tests for image quality assessment."""

import pytest
import numpy as np
import cv2
from ocr_pipeline.quality.image_quality import ImageQualityAssessor, assess_image_quality


@pytest.fixture
def quality_config():
    """Sample quality configuration."""
    return {
        'min_blur_score': 100.0,
        'min_brightness': 30,
        'max_brightness': 225,
        'min_contrast_ratio': 1.5,
        'min_edge_density': 0.01,
        'min_resolution': 300,
        'weights': {
            'blur': 0.35,
            'brightness': 0.20,
            'resolution': 0.25,
            'contrast': 0.20
        }
    }


@pytest.fixture
def good_image():
    """Create a good quality test image."""
    # Create a sharp, well-lit image with text-like patterns
    image = np.ones((800, 600, 3), dtype=np.uint8) * 200
    
    # Add some text-like patterns
    for i in range(10):
        cv2.rectangle(image, (50 + i*50, 100), (80 + i*50, 150), (0, 0, 0), 2)
    
    return image


@pytest.fixture
def blurry_image():
    """Create a blurry test image."""
    image = np.ones((800, 600, 3), dtype=np.uint8) * 200
    # Apply heavy blur
    return cv2.GaussianBlur(image, (51, 51), 0)


@pytest.fixture
def dark_image():
    """Create a dark test image."""
    return np.ones((800, 600, 3), dtype=np.uint8) * 20


def test_blur_detection(quality_config, good_image, blurry_image):
    """Test blur score calculation."""
    assessor = ImageQualityAssessor(quality_config)
    
    good_blur = assessor.calculate_blur_score(good_image)
    blurry_blur = assessor.calculate_blur_score(blurry_image)
    
    assert good_blur > blurry_blur
    assert good_blur > quality_config['min_blur_score']


def test_brightness_detection(quality_config, good_image, dark_image):
    """Test brightness calculation."""
    assessor = ImageQualityAssessor(quality_config)
    
    good_brightness = assessor.calculate_brightness_score(good_image)
    dark_brightness = assessor.calculate_brightness_score(dark_image)
    
    assert good_brightness > dark_brightness
    assert good_brightness > quality_config['min_brightness']
    assert dark_brightness < quality_config['min_brightness']


def test_quality_gate_pass(quality_config, good_image):
    """Test that good image passes quality gate."""
    metrics = assess_image_quality(good_image, quality_config)
    
    assert metrics.passed is True
    assert metrics.composite_score > 0.5
    assert len(metrics.failure_reasons) == 0


def test_quality_gate_fail(quality_config, blurry_image):
    """Test that blurry image fails quality gate."""
    metrics = assess_image_quality(blurry_image, quality_config)
    
    assert metrics.passed is False
    assert len(metrics.failure_reasons) > 0


def test_resolution_score(quality_config):
    """Test resolution scoring."""
    assessor = ImageQualityAssessor(quality_config)
    
    # Low resolution image
    low_res = np.ones((100, 100, 3), dtype=np.uint8)
    low_res_score = assessor.calculate_resolution_score(low_res)
    
    # High resolution image
    high_res = np.ones((2000, 1500, 3), dtype=np.uint8)
    high_res_score = assessor.calculate_resolution_score(high_res)
    
    assert high_res_score > low_res_score


def test_contrast_score(quality_config):
    """Test contrast calculation."""
    assessor = ImageQualityAssessor(quality_config)
    
    # Low contrast (uniform gray)
    low_contrast = np.ones((800, 600, 3), dtype=np.uint8) * 128
    low_score = assessor.calculate_contrast_score(low_contrast)
    
    # High contrast (black and white)
    high_contrast = np.zeros((800, 600, 3), dtype=np.uint8)
    high_contrast[:, :300] = 255
    high_score = assessor.calculate_contrast_score(high_contrast)
    
    assert high_score > low_score


def test_edge_density(quality_config, good_image):
    """Test edge density calculation."""
    assessor = ImageQualityAssessor(quality_config)
    
    edge_density = assessor.calculate_edge_density(good_image)
    
    assert 0.0 <= edge_density <= 1.0
    assert edge_density > quality_config['min_edge_density']
