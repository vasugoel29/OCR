#!/usr/bin/env python3
"""Test script to demonstrate multi-document detection."""

import cv2
import numpy as np
from src.segmentation import SegmentationPipeline

def create_multi_document_test_image():
    """Create a synthetic image with 2 documents side by side."""
    # Create white background
    image = np.ones((1000, 2000, 3), dtype=np.uint8) * 255
    
    # Document 1 (left side) - simulate a document with border
    cv2.rectangle(image, (50, 100), (900, 900), (0, 0, 0), 3)
    cv2.putText(image, "INVOICE #001", (100, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    cv2.putText(image, "Total: $500", (100, 400), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Document 2 (right side) - simulate another document with border
    cv2.rectangle(image, (1100, 100), (1950, 900), (0, 0, 0), 3)
    cv2.putText(image, "RECEIPT #002", (1150, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    cv2.putText(image, "Amount: $300", (1150, 400), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    return image

def test_single_document():
    """Test with single document."""
    print("\n" + "="*60)
    print("TEST 1: Single Document")
    print("="*60)
    
    # Create single document image
    image = np.ones((1000, 800, 3), dtype=np.uint8) * 255
    cv2.rectangle(image, (50, 50), (750, 950), (0, 0, 0), 3)
    cv2.putText(image, "INVOICE", (100, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    
    # Save for inspection
    cv2.imwrite('test_single_doc.jpg', image)
    
    # Detect regions
    config = {'enabled': True}
    pipeline = SegmentationPipeline(config)
    regions = pipeline.detect_regions(image)
    
    print(f"✓ Regions detected: {len(regions)}")
    for i, region in enumerate(regions):
        print(f"  Region {i+1}:")
        print(f"    - Method: {region.detection_method}")
        print(f"    - Confidence: {region.confidence:.3f}")
        print(f"    - BBox: ({region.bbox.x}, {region.bbox.y}, {region.bbox.width}, {region.bbox.height})")
        print(f"    - Area ratio: {region.area_ratio:.3f}")

def test_multi_document():
    """Test with multiple documents."""
    print("\n" + "="*60)
    print("TEST 2: Multiple Documents")
    print("="*60)
    
    # Create multi-document image
    image = create_multi_document_test_image()
    
    # Save for inspection
    cv2.imwrite('test_multi_doc.jpg', image)
    
    # Detect regions
    config = {'enabled': True}
    pipeline = SegmentationPipeline(config)
    regions = pipeline.detect_regions(image)
    
    print(f"✓ Regions detected: {len(regions)}")
    for i, region in enumerate(regions):
        print(f"  Region {i+1}:")
        print(f"    - Method: {region.detection_method}")
        print(f"    - Confidence: {region.confidence:.3f}")
        print(f"    - BBox: ({region.bbox.x}, {region.bbox.y}, {region.bbox.width}, {region.bbox.height})")
        print(f"    - Area ratio: {region.area_ratio:.3f}")
    
    if len(regions) > 1:
        print(f"\n✅ SUCCESS: Detected {len(regions)} separate document regions!")
    else:
        print(f"\n⚠️  Only detected {len(regions)} region (expected 2)")

def test_overlapping_documents():
    """Test with overlapping documents."""
    print("\n" + "="*60)
    print("TEST 3: Overlapping Documents")
    print("="*60)
    
    # Create image with overlapping documents
    image = np.ones((1000, 1500, 3), dtype=np.uint8) * 255
    
    # Document 1 (background)
    cv2.rectangle(image, (100, 100), (900, 700), (0, 0, 0), 3)
    cv2.putText(image, "DOC 1", (200, 300), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    
    # Document 2 (overlapping)
    cv2.rectangle(image, (600, 400), (1400, 900), (0, 0, 0), 3)
    cv2.putText(image, "DOC 2", (700, 600), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    
    # Save for inspection
    cv2.imwrite('test_overlapping_docs.jpg', image)
    
    # Detect regions
    config = {'enabled': True}
    pipeline = SegmentationPipeline(config)
    regions = pipeline.detect_regions(image)
    
    print(f"✓ Regions detected: {len(regions)}")
    for i, region in enumerate(regions):
        print(f"  Region {i+1}:")
        print(f"    - Method: {region.detection_method}")
        print(f"    - Confidence: {region.confidence:.3f}")
        print(f"    - BBox: ({region.bbox.x}, {region.bbox.y}, {region.bbox.width}, {region.bbox.height})")
        print(f"    - Area ratio: {region.area_ratio:.3f}")

def test_with_pipeline():
    """Test multi-document with full OCR pipeline."""
    print("\n" + "="*60)
    print("TEST 4: Full Pipeline with Multi-Document")
    print("="*60)
    
    from src.pipeline import OCRPipeline
    
    # Create multi-document image
    image = create_multi_document_test_image()
    cv2.imwrite('test_pipeline_multi.jpg', image)
    
    # Process with pipeline
    pipeline = OCRPipeline()
    result = pipeline.process_document('test_pipeline_multi.jpg', document_type='invoice')
    
    print(f"✓ Regions detected: {result.regions_detected}")
    print(f"✓ Multi-document flag: {result.multi_document_flag}")
    print(f"✓ Selected region: {result.region_selected}")
    print(f"✓ Decision: {result.decision}")
    print(f"✓ Spatial compactness: {result.confidence.spatial_compactness_score:.3f}")

if __name__ == '__main__':
    print("\n" + "="*60)
    print("MULTI-DOCUMENT SEGMENTATION TEST SUITE")
    print("="*60)
    
    try:
        test_single_document()
        test_multi_document()
        test_overlapping_documents()
        test_with_pipeline()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETED")
        print("="*60)
        print("\nGenerated test images:")
        print("  - test_single_doc.jpg")
        print("  - test_multi_doc.jpg")
        print("  - test_overlapping_docs.jpg")
        print("  - test_pipeline_multi.jpg")
        print("\nCheck these images to see the segmentation results!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
