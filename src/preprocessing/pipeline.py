"""Preprocessing pipeline orchestrator."""

import cv2
import numpy as np
from typing import Dict, List, Optional
from .corrections import ImageCorrector


class PreprocessingPipeline:
    """Orchestrates image preprocessing steps."""
    
    def __init__(self, config: Dict):
        """Initialize preprocessing pipeline.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config
        self.corrector = ImageCorrector(config)
        self.steps_applied = []
    
    def process(self, image: np.ndarray, save_intermediates: bool = False) -> Dict:
        """Apply full preprocessing pipeline.
        
        Args:
            image: Input image
            save_intermediates: Whether to save intermediate results
            
        Returns:
            Dictionary with processed image and metadata
        """
        self.steps_applied = []
        intermediates = {}
        
        current_image = image.copy()
        
        if save_intermediates:
            intermediates['original'] = image.copy()
        
        # Step 1: Noise removal (do this first to help other algorithms)
        if self.config.get('enable_noise_removal', True):
            current_image = self.corrector.remove_noise(current_image)
            self.steps_applied.append('noise_removal')
            if save_intermediates:
                intermediates['denoised'] = current_image.copy()
        
        # Step 2: Skew correction
        if self.config.get('enable_skew_correction', True):
            current_image = self.corrector.correct_skew(current_image)
            self.steps_applied.append('skew_correction')
            if save_intermediates:
                intermediates['deskewed'] = current_image.copy()
        
        # Step 3: Perspective correction
        if self.config.get('enable_perspective_correction', True):
            current_image = self.corrector.correct_perspective(current_image)
            self.steps_applied.append('perspective_correction')
            if save_intermediates:
                intermediates['perspective_corrected'] = current_image.copy()
        
        # Step 4: Illumination normalization
        if self.config.get('enable_illumination_normalization', True):
            current_image = self.corrector.normalize_illumination(current_image)
            self.steps_applied.append('illumination_normalization')
            if save_intermediates:
                intermediates['illumination_normalized'] = current_image.copy()
        
        return {
            'processed_image': current_image,
            'steps_applied': self.steps_applied,
            'intermediates': intermediates
        }
    
    def process_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Process image specifically for OCR (includes binarization).
        
        Args:
            image: Input image
            
        Returns:
            Processed and binarized image ready for OCR
        """
        # Apply standard preprocessing
        result = self.process(image)
        processed = result['processed_image']
        
        # Apply adaptive thresholding for better OCR
        binary = self.corrector.apply_adaptive_threshold(processed)
        
        return binary
    
    def get_steps_applied(self) -> List[str]:
        """Get list of preprocessing steps that were applied.
        
        Returns:
            List of step names
        """
        return self.steps_applied.copy()
