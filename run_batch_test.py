#!/usr/bin/env python3
"""
Script to run OCR pipeline on all images in tests/images.
"""

import os
import sys
import glob
from pathlib import Path
import logging

# Ensure src is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import OCRPipeline

def main():
    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(base_dir, 'tests', 'images')
    
    if not os.path.exists(images_dir):
        print(f"Error: Directory not found: {images_dir}")
        print("Please ensure 'tests/images' exists and contains images.")
        return
    
    # Get all images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(images_dir, ext)))
        # Case insensitive check (e.g. JPG, PNG)
        image_paths.extend(glob.glob(os.path.join(images_dir, ext.upper())))
    
    # Remove duplicates
    image_paths = sorted(list(set(image_paths)))
    
    if not image_paths:
        print(f"No images found in {images_dir}")
        return
    
    print(f"Found {len(image_paths)} images. Starting processing...")
    print("-" * 60)
    
    # Initialize pipeline
    try:
        pipeline = OCRPipeline()
        # Set logging to INFO to see progress
        pipeline.logger.setLevel(logging.INFO)
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        import traceback
        traceback.print_exc()
        return

    # Process images
    results_summary = []
    
    for i, img_path in enumerate(image_paths):
        filename = os.path.basename(img_path)
        print(f"[{i+1}/{len(image_paths)}] Processing {filename}...")
        
        try:
            # Run pipeline
            result = pipeline.process_document(img_path, document_type='auto')
            
            # Store summary
            status = "SUCCESS" if result.decision != 'error' else "FAILED"
            
            # Get primary reason for decision
            reason = "-"
            if hasattr(result, 'decision_details') and result.decision_details:
                if 'reasons' in result.decision_details and result.decision_details['reasons']:
                    reason = result.decision_details['reasons'][0]
            
            summary = {
                'file': filename,
                'status': status,
                'type': result.document_type,
                'decision': result.decision,
                'score': f"{result.confidence.final_score:.2f}" if hasattr(result.confidence, 'final_score') else "0.00",
                'extracted': len(result.extracted_fields),
                'reason': reason
            }
            results_summary.append(summary)
            
            # Print result for this file
            print(f"  Type: {result.document_type}")
            print(f"  Decision: {result.decision} (Score: {summary['score']})")
            print(f"  Reason: {reason}")
            print(f"  Extracted Fields ({len(result.extracted_fields)}):")
            for k, v in result.extracted_fields.items():
                print(f"    - {k}: {v}")
            print("-" * 60)
            
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            results_summary.append({
                'file': filename,
                'status': "ERROR",
                'error': str(e),
                'reason': str(e)
            })
            print("-" * 60)

    # Print final summary
    print("\n" + "=" * 100)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 100)
    print(f"{'Filename':<25} | {'Type':<12} | {'Decision':<10} | {'Score':<6} | {'Fields':<6} | {'Reason':<30}")
    print("-" * 100)
    
    for r in results_summary:
        if r.get('status') == 'ERROR':
            print(f"{r['file']:<25} | {'ERROR':<12} | {'-':<10} | {'-':<6} | {'-':<6} | {r.get('error', '')[0:29]:<30}")
        else:
            reason_trunc = r['reason'][:29] if r['reason'] else "-"
            print(f"{r['file']:<25} | {r['type']:<12} | {r['decision']:<10} | {r['score']:<6} | {r['extracted']:<6} | {reason_trunc:<30}")
            
    print("=" * 100)

if __name__ == "__main__":
    main()
