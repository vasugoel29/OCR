"""Segmentation pipeline orchestrator for document detection."""

import logging
from typing import List, Optional
import numpy as np

from .region import Region, BoundingBox
from .document_detector import DocumentDetector
from .text_clustering import TextClusterer


class SegmentationPipeline:
    """Orchestrate document detection and region extraction."""
    
    def __init__(self, config: dict = None):
        """Initialize segmentation pipeline.
        
        Args:
            config: Configuration dictionary for segmentation
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Check if segmentation is enabled
        self.enabled = self.config.get('enabled', True)
        
        # Initialize detectors
        self.document_detector = DocumentDetector(self.config)
        self.text_clusterer = TextClusterer(self.config)
        
        # Get configuration parameters
        self.max_regions = self.config.get('max_regions', 5)
        self.min_region_confidence = self.config.get('min_region_confidence', 0.3)
        
        # Detection method flags
        contour_config = self.config.get('contour_detection', {})
        text_config = self.config.get('text_clustering', {})
        self.use_contour_detection = contour_config.get('enabled', True)
        self.use_text_clustering = text_config.get('enabled', True)
    
    def detect_regions(self, image: np.ndarray, 
                      ocr_boxes: Optional[List] = None) -> List[Region]:
        """Detect document regions in an image.
        
        Args:
            image: Input image
            ocr_boxes: Optional list of OCR bounding boxes for text clustering
            
        Returns:
            List of detected regions, sorted by confidence (highest first)
        """
        if not self.enabled:
            self.logger.debug("Segmentation disabled, returning full image as single region")
            return self._create_full_image_region(image)
        
        self.logger.info("Starting document region detection")
        
        all_regions = []
        
        # Method 1: Contour-based detection
        if self.use_contour_detection:
            try:
                contour_regions = self.document_detector.detect_contours(image)
                all_regions.extend(contour_regions)
                self.logger.debug(f"Contour detection found {len(contour_regions)} regions")
            except Exception as e:
                self.logger.warning(f"Contour detection failed: {e}")
        
        # Method 2: Text-density clustering
        if self.use_text_clustering and ocr_boxes:
            try:
                text_regions = self.text_clusterer.cluster_text_regions(
                    ocr_boxes, 
                    image.shape
                )
                # Extract actual image regions
                for region in text_regions:
                    region.image = self._extract_region_image(image, region.bbox)
                all_regions.extend(text_regions)
                self.logger.debug(f"Text clustering found {len(text_regions)} regions")
            except Exception as e:
                self.logger.warning(f"Text clustering failed: {e}")
        
        # If no regions detected, return full image
        if not all_regions:
            self.logger.info("No regions detected, using full image")
            return self._create_full_image_region(image)
        
        # Deduplicate overlapping regions
        regions = self._deduplicate_regions(all_regions)
        
        # Filter by confidence
        regions = self._filter_regions(regions)
        
        # Sort by confidence (highest first)
        regions.sort(key=lambda r: r.confidence, reverse=True)
        
        # Limit number of regions
        if len(regions) > self.max_regions:
            self.logger.warning(f"Too many regions ({len(regions)}), limiting to {self.max_regions}")
            regions = regions[:self.max_regions]
        
        self.logger.info(f"Final region count: {len(regions)}")
        
        return regions
    
    def _create_full_image_region(self, image: np.ndarray) -> List[Region]:
        """Create a single region representing the full image.
        
        Args:
            image: Input image
            
        Returns:
            List containing single region for full image
        """
        h, w = image.shape[:2]
        return [Region(
            bbox=BoundingBox(0, 0, w, h),
            image=image.copy(),
            confidence=1.0,
            detection_method='full_image',
            area_ratio=1.0
        )]
    
    def _extract_region_image(self, image: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        """Extract region image from source image.
        
        Args:
            image: Source image
            bbox: Bounding box of region
            
        Returns:
            Extracted region image
        """
        return image[bbox.y:bbox.y+bbox.height, bbox.x:bbox.x+bbox.width].copy()
    
    def _deduplicate_regions(self, regions: List[Region]) -> List[Region]:
        """Remove duplicate or heavily overlapping regions.
        
        Args:
            regions: List of regions to deduplicate
            
        Returns:
            Deduplicated list of regions
        """
        if len(regions) <= 1:
            return regions
        
        # Sort by confidence (keep higher confidence regions)
        sorted_regions = sorted(regions, key=lambda r: r.confidence, reverse=True)
        
        unique_regions = []
        
        for region in sorted_regions:
            # Check if this region overlaps significantly with any kept region
            is_duplicate = False
            for kept_region in unique_regions:
                if region.bbox.overlaps_with(kept_region.bbox, threshold=0.7):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_regions.append(region)
        
        self.logger.debug(f"Deduplicated {len(regions)} regions to {len(unique_regions)}")
        return unique_regions
    
    def _filter_regions(self, regions: List[Region]) -> List[Region]:
        """Filter regions by quality criteria.
        
        Args:
            regions: List of regions to filter
            
        Returns:
            Filtered list of regions
        """
        filtered = []
        
        for region in regions:
            # Filter by confidence
            if region.confidence < self.min_region_confidence:
                self.logger.debug(f"Filtering region with low confidence: {region.confidence:.3f}")
                continue
            
            # Filter by area (already done in detectors, but double-check)
            if region.area_ratio < 0.05:  # Too small
                self.logger.debug(f"Filtering region with small area: {region.area_ratio:.3f}")
                continue
            
            # Filter by aspect ratio (sanity check)
            aspect_ratio = region.bbox.aspect_ratio
            if aspect_ratio < 0.1 or aspect_ratio > 10.0:  # Extreme aspect ratios
                self.logger.debug(f"Filtering region with extreme aspect ratio: {aspect_ratio:.3f}")
                continue
            
            filtered.append(region)
        
        self.logger.debug(f"Filtered {len(regions)} regions to {len(filtered)}")
        return filtered
