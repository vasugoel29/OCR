"""Pytest configuration."""

import pytest
import sys
from pathlib import Path

# Add ocr_pipeline to Python path
package_path = Path(__file__).parent
sys.path.insert(0, str(package_path))
