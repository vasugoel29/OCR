"""Data models for OCR results."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class WordData:
    """Data for a single OCR word."""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    line_num: int
    word_num: int
    is_numeric: bool = False
    is_stopword: bool = False


@dataclass
class LineData:
    """Data for a single OCR line."""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    words: List[WordData] = field(default_factory=list)


@dataclass
class OCRResult:
    """Complete OCR result for a document."""
    full_text: str
    mean_confidence: float
    words: List[WordData] = field(default_factory=list)
    lines: List[LineData] = field(default_factory=list)
    total_words: int = 0
    low_confidence_words: int = 0
    numeric_words: int = 0
    
    def get_words_by_confidence(self, min_confidence: float) -> List[WordData]:
        """Get words above a confidence threshold.
        
        Args:
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of words above threshold
        """
        return [w for w in self.words if w.confidence >= min_confidence]
    
    def get_numeric_words(self) -> List[WordData]:
        """Get all numeric words.
        
        Returns:
            List of numeric words
        """
        return [w for w in self.words if w.is_numeric]
    
    def get_text_by_region(self, region: Tuple[int, int, int, int]) -> str:
        """Get text within a specific region.
        
        Args:
            region: (x, y, width, height) bounding box
            
        Returns:
            Concatenated text from region
        """
        x, y, w, h = region
        words_in_region = []
        
        for word in self.words:
            wx, wy, ww, wh = word.bbox
            # Check if word center is in region
            word_center_x = wx + ww // 2
            word_center_y = wy + wh // 2
            
            if (x <= word_center_x <= x + w and 
                y <= word_center_y <= y + h):
                words_in_region.append(word.text)
        
        return ' '.join(words_in_region)
