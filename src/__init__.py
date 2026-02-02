"""OCR Pipeline Package."""

from .pipeline import OCRPipeline, PipelineResult
from .utils import load_config, setup_logging

__version__ = '1.0.0'
__all__ = ['OCRPipeline', 'PipelineResult', 'load_config', 'setup_logging']
