"""OCR Pipeline - Production-ready OCR system for Indian identity documents."""

import sys
import os

# Python 3.13+ compatibility: Make imghdr available before any imports
# This must happen before paddleocr is imported
if sys.version_info >= (3, 13):
    try:
        import imghdr
    except ImportError:
        # Add compat directory to path and import our shim
        compat_path = os.path.join(os.path.dirname(__file__), 'compat')
        if compat_path not in sys.path:
            sys.path.insert(0, compat_path)
        # Import and register in sys.modules so paddleocr can find it
        from .compat import imghdr
        sys.modules['imghdr'] = imghdr

__version__ = "1.0.0"

from .core.pipeline import OCRPipeline, PipelineResult
from .utils import load_config, setup_logging

__all__ = ["OCRPipeline", "PipelineResult", "load_config", "setup_logging"]
