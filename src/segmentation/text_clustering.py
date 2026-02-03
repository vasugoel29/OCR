"""Text-density based region clustering for document detection."""

import logging
from typing import List, Tuple
import numpy as np
from sklearn.cluster import DBSCAN

from .region import Region, BoundingBox


class TextClusterer:
    """Detect document regions based on text density clustering."""
    
    def __init__(self, config: dict = None):
        """Initialize text clusterer.
        
        Args:
            config: Configuration dictionary with clustering parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Get configuration parameters
        text_config = self.config.get('text_clustering', {})
        self.min_cluster_size = text_config.get('min_cluster_size', 5)
        self.eps = text_config.get('eps', 50)
        self.min_samples = text_config.get('min_samples', 3)
    
    def cluster_text_regions(self, ocr_boxes: List[Tuple], 
                            image_shape: Tuple) -> List[Region]:
        """Cluster text bounding boxes into document regions.
        
        Args:
            ocr_boxes: List of (x, y, w, h) bounding boxes from OCR
            image_shape: Shape of the image (height, width, ...)
            
        Returns:
            List of detected regions based on text clustering
        """
        if not ocr_boxes or len(ocr_boxes) < self.min_cluster_size:
            self.logger.debug("Not enough text boxes for clustering")
            return []
        
        self.logger.debug(f"Clustering {len(ocr_boxes)} text boxes")
        
        # Extract centers of bounding boxes
        centers = []
        for box in ocr_boxes:
            if len(box) == 4:
                x, y, w, h = box
                centers.append([x + w/2, y + h/2])
            else:
                self.logger.warning(f"Invalid box format: {box}")
        
        if len(centers) < self.min_cluster_size:
            return []
        
        centers = np.array(centers)
        
        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = clustering.fit_predict(centers)
        
        # Group boxes by cluster
        clusters = {}
        for idx, label in enumerate(labels):
            if label == -1:  # Noise point
                continue
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(ocr_boxes[idx])
        
        self.logger.debug(f"Found {len(clusters)} text clusters")
        
        # Create regions from clusters
        regions = []
        image_area = image_shape[0] * image_shape[1]
        
        for cluster_id, boxes in clusters.items():
            if len(boxes) >= self.min_cluster_size:
                region = self._create_region_from_cluster(boxes, image_shape, image_area)
                if region:
                    regions.append(region)
        
        # Merge overlapping clusters
        regions = self._merge_overlapping_clusters(regions)
        
        self.logger.info(f"Detected {len(regions)} regions from text clustering")
        return regions
    
    def _create_region_from_cluster(self, boxes: List[Tuple], 
                                    image_shape: Tuple,
                                    image_area: int) -> Region:
        """Create a region from a cluster of text boxes.
        
        Args:
            boxes: List of bounding boxes in the cluster
            image_shape: Shape of the image
            image_area: Total image area
            
        Returns:
            Region object or None if invalid
        """
        try:
            # Calculate bounding box encompassing all text boxes
            min_x = min(box[0] for box in boxes)
            min_y = min(box[1] for box in boxes)
            max_x = max(box[0] + box[2] for box in boxes)
            max_y = max(box[1] + box[3] for box in boxes)
            
            # Add padding around text cluster
            padding = 20  # pixels
            min_x = max(0, min_x - padding)
            min_y = max(0, min_y - padding)
            max_x = min(image_shape[1], max_x + padding)
            max_y = min(image_shape[0], max_y + padding)
            
            width = max_x - min_x
            height = max_y - min_y
            
            # Validate region
            if width <= 0 or height <= 0:
                return None
            
            area = width * height
            area_ratio = area / image_area
            
            # Calculate confidence based on cluster density
            # More boxes in cluster = higher confidence
            text_density = len(boxes) / area if area > 0 else 0
            confidence = min(1.0, text_density * 1000)  # Normalize
            
            # Create dummy image (will be extracted from source later)
            region_image = np.zeros((height, width, 3), dtype=np.uint8)
            
            return Region(
                bbox=BoundingBox(int(min_x), int(min_y), int(width), int(height)),
                image=region_image,
                confidence=confidence,
                detection_method='text_cluster',
                area_ratio=area_ratio
            )
        
        except Exception as e:
            self.logger.warning(f"Failed to create region from cluster: {e}")
            return None
    
    def _merge_overlapping_clusters(self, regions: List[Region]) -> List[Region]:
        """Merge overlapping region clusters.
        
        Args:
            regions: List of regions to merge
            
        Returns:
            List of merged regions
        """
        if len(regions) <= 1:
            return regions
        
        merged = []
        used = set()
        
        for i, region1 in enumerate(regions):
            if i in used:
                continue
            
            # Find all regions that overlap with this one
            overlapping = [region1]
            for j, region2 in enumerate(regions[i+1:], start=i+1):
                if j in used:
                    continue
                
                if region1.bbox.overlaps_with(region2.bbox, threshold=0.3):
                    overlapping.append(region2)
                    used.add(j)
            
            # Merge overlapping regions
            if len(overlapping) > 1:
                merged_region = self._merge_regions(overlapping)
                if merged_region:
                    merged.append(merged_region)
            else:
                merged.append(region1)
        
        return merged
    
    def _merge_regions(self, regions: List[Region]) -> Region:
        """Merge multiple regions into one.
        
        Args:
            regions: List of regions to merge
            
        Returns:
            Merged region
        """
        # Calculate bounding box encompassing all regions
        min_x = min(r.bbox.x for r in regions)
        min_y = min(r.bbox.y for r in regions)
        max_x = max(r.bbox.x + r.bbox.width for r in regions)
        max_y = max(r.bbox.y + r.bbox.height for r in regions)
        
        width = max_x - min_x
        height = max_y - min_y
        
        # Average confidence
        avg_confidence = sum(r.confidence for r in regions) / len(regions)
        
        # Average area ratio
        avg_area_ratio = sum(r.area_ratio for r in regions) / len(regions)
        
        # Create dummy image
        region_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        return Region(
            bbox=BoundingBox(min_x, min_y, width, height),
            image=region_image,
            confidence=avg_confidence,
            detection_method='merged',
            area_ratio=avg_area_ratio
        )
