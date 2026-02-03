"""OCR module."""

from .models import OCRResult, WordData, LineData
from .engine import PaddleOCREngine, extract_text_from_image

__all__ = ['OCRResult', 'WordData', 'LineData', 'PaddleOCREngine', 'extract_text_from_image']
