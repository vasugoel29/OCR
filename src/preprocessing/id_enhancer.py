"""Enhanced preprocessing specifically for Indian ID documents (Aadhaar, PAN, etc.)."""

import cv2
import numpy as np
from typing import Tuple, Optional
from .corrections import ImageCorrector

class IDDocumentEnhancer:
    """Enhanced preprocessing for ID documents."""
    
    def __init__(self):
        """Initialize ID document enhancer."""
        # Initialize corrector with default config for deskewing
        self.corrector = ImageCorrector({
            'enable_skew_correction': True,
            'max_skew_angle': 45
        })
    
    def enhance_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Apply comprehensive enhancement pipeline for ID documents.
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image optimized for OCR
        """
        # Step 1: Resize for better detail (increased resolution)
        image = self._resize_if_needed(image, min_width=1600)
        
        # Step 1.5: Deskew
        image = self.deskew_document(image)
        
        # Step 2: Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Step 3: Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Step 4: Increase contrast with CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Step 5: Sharpen
        sharpened = self._sharpen_image(enhanced)
        
        # Step 6: Adaptive thresholding for better text separation
        binary = cv2.adaptiveThreshold(
            sharpened,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            15,
            10
        )
        
        # Step 7: Morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned

    def deskew_document(self, image: np.ndarray) -> np.ndarray:
        """Deskew document using ImageCorrector (Hough Lines).
        
        Args:
            image: Input image
            
        Returns:
            Deskewed image
        """
        return self.corrector.correct_skew(image)
    
    def _resize_if_needed(self, image: np.ndarray, min_width: int = 1600) -> np.ndarray:
        """Resize image if it's too small.
        
        Args:
            image: Input image
            min_width: Minimum width in pixels
            
        Returns:
            Resized image
        """
        height, width = image.shape[:2]
        
        if width < min_width:
            scale = min_width / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            return resized
        
        return image
    
    def _sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """Sharpen image to enhance text edges.
        
        Args:
            image: Input image
            
        Returns:
            Sharpened image
        """
        # Sharpening kernel
        kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ])
        
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened
