"""OCR engine integration using Tesseract (Approach 2)."""

import cv2
import numpy as np
import pytesseract
from typing import Dict, List, Set
import re
from .models import OCRResult, WordData, LineData


class TesseractEngine:
    """Tesseract OCR engine wrapper with confidence extraction."""
    
    def __init__(self, config: Dict):
        """Initialize Tesseract engine.
        
        Args:
            config: OCR configuration dictionary
        """
        self.config = config
        self.language = config.get('language', 'eng')  # Default to English
        self.tesseract_config = config.get('tesseract_config', '--oem 3 --psm 3')
        self.min_word_confidence = config.get('min_word_confidence', 40)
        self.min_words_detected = config.get('min_words_detected', 5)
        
        # Load stopwords
        self.stopwords = set(config.get('stopwords', []))
        
        # Confidence weights
        self.numeric_weight = config.get('numeric_token_weight', 1.5)
        self.alpha_weight = config.get('alpha_token_weight', 1.0)
        self.stopword_weight = config.get('stopword_weight', 0.3)
    
    def extract_text(self, image: np.ndarray) -> OCRResult:
        """Extract text from image with word-level confidence.
        
        Args:
            image: Input image (preprocessed)
            
        Returns:
            OCRResult object with detailed information
        """
        # Get detailed OCR data
        ocr_data = pytesseract.image_to_data(
            image,
            lang=self.language,
            config=self.tesseract_config,
            output_type=pytesseract.Output.DICT
        )
        
        # Parse OCR data
        words = []
        lines_dict = {}
        
        n_boxes = len(ocr_data['text'])
        for i in range(n_boxes):
            text = ocr_data['text'][i].strip()
            
            # Skip empty text
            if not text:
                continue
            
            confidence = float(ocr_data['conf'][i])
            
            # Skip very low confidence (likely noise)
            if confidence < 0:
                continue
            
            # Extract bounding box
            x = ocr_data['left'][i]
            y = ocr_data['top'][i]
            w = ocr_data['width'][i]
            h = ocr_data['height'][i]
            bbox = (x, y, w, h)
            
            # Get line and word numbers
            line_num = ocr_data['line_num'][i]
            word_num = ocr_data['word_num'][i]
            
            # Determine if numeric
            is_numeric = self._is_numeric(text)
            
            # Determine if stopword
            is_stopword = text.lower() in self.stopwords
            
            # Create word data
            word_data = WordData(
                text=text,
                confidence=confidence,
                bbox=bbox,
                line_num=line_num,
                word_num=word_num,
                is_numeric=is_numeric,
                is_stopword=is_stopword
            )
            
            words.append(word_data)
            
            # Group by line
            if line_num not in lines_dict:
                lines_dict[line_num] = []
            lines_dict[line_num].append(word_data)
        
        # Create line data
        lines = []
        for line_num in sorted(lines_dict.keys()):
            line_words = lines_dict[line_num]
            line_text = ' '.join([w.text for w in line_words])
            
            # Calculate line confidence
            line_confidence = np.mean([w.confidence for w in line_words])
            
            # Calculate line bounding box
            if line_words:
                min_x = min(w.bbox[0] for w in line_words)
                min_y = min(w.bbox[1] for w in line_words)
                max_x = max(w.bbox[0] + w.bbox[2] for w in line_words)
                max_y = max(w.bbox[1] + w.bbox[3] for w in line_words)
                line_bbox = (min_x, min_y, max_x - min_x, max_y - min_y)
            else:
                line_bbox = (0, 0, 0, 0)
            
            line_data = LineData(
                text=line_text,
                confidence=line_confidence,
                bbox=line_bbox,
                words=line_words
            )
            lines.append(line_data)
        
        # Calculate overall statistics
        full_text = '\n'.join([line.text for line in lines])
        
        if words:
            # Calculate weighted mean confidence
            mean_confidence = self._calculate_weighted_confidence(words)
            
            # Count low confidence words
            low_confidence_words = sum(1 for w in words if w.confidence < self.min_word_confidence)
            
            # Count numeric words
            numeric_words = sum(1 for w in words if w.is_numeric)
        else:
            mean_confidence = 0.0
            low_confidence_words = 0
            numeric_words = 0
        
        return OCRResult(
            full_text=full_text,
            mean_confidence=mean_confidence,
            words=words,
            lines=lines,
            total_words=len(words),
            low_confidence_words=low_confidence_words,
            numeric_words=numeric_words
        )
    
    def calculate_ocr_confidence_score(self, ocr_result: OCRResult) -> float:
        """Calculate normalized OCR confidence score.
        
        Args:
            ocr_result: OCR result object
            
        Returns:
            Confidence score normalized to [0, 1]
        """
        if ocr_result.total_words == 0:
            return 0.0
        
        # Check minimum words threshold
        if ocr_result.total_words < self.min_words_detected:
            return 0.0
        
        # Penalize high percentage of low confidence words
        low_conf_ratio = ocr_result.low_confidence_words / ocr_result.total_words
        if low_conf_ratio > 0.4:
            return 0.0
        
        # Normalize mean confidence to [0, 1]
        normalized_confidence = ocr_result.mean_confidence / 100.0
        
        # Bonus for numeric content (important for invoices)
        numeric_ratio = ocr_result.numeric_words / ocr_result.total_words
        numeric_bonus = min(0.1, numeric_ratio * 0.2)
        
        final_score = min(1.0, normalized_confidence + numeric_bonus)
        
        return final_score
    
    def _calculate_weighted_confidence(self, words: List[WordData]) -> float:
        """Calculate weighted average confidence.
        
        Args:
            words: List of word data
            
        Returns:
            Weighted mean confidence
        """
        if not words:
            return 0.0
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for word in words:
            # Determine weight based on word type
            if word.is_stopword:
                weight = self.stopword_weight
            elif word.is_numeric:
                weight = self.numeric_weight
            else:
                weight = self.alpha_weight
            
            weighted_sum += word.confidence * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight
    
    def _is_numeric(self, text: str) -> bool:
        """Check if text is primarily numeric.
        
        Args:
            text: Text to check
            
        Returns:
            True if text contains numbers
        """
        # Remove common separators
        cleaned = re.sub(r'[,.\s$€£¥]', '', text)
        
        # Check if majority is digits
        if not cleaned:
            return False
        
        digit_count = sum(c.isdigit() for c in cleaned)
        return digit_count / len(cleaned) > 0.5


def extract_text_from_image(image: np.ndarray, config: Dict) -> OCRResult:
    """Convenience function to extract text from image.
    
    Args:
        image: Input image
        config: OCR configuration
        
    Returns:
        OCRResult object
    """
    engine = TesseractEngine(config)
    return engine.extract_text(image)
