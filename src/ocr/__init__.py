"""OCR module."""

from .models import OCRResult, WordData, LineData
from .engine import TesseractEngine, extract_text_from_image

__all__ = ['OCRResult', 'WordData', 'LineData', 'TesseractEngine', 'extract_text_from_image']
