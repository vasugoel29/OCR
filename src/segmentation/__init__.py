"""Document segmentation module for multi-document handling.

Provides document detection and region extraction capabilities.
"""

from .region import Region, BoundingBox
from .document_detector import DocumentDetector
from .text_clustering import TextClusterer
from .segmentation_pipeline import SegmentationPipeline

__all__ = [
    'Region',
    'BoundingBox',
    'DocumentDetector',
    'TextClusterer',
    'SegmentationPipeline'
]
