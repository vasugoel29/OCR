"""Example: Process a single document."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pipeline import OCRPipeline
import json


def main():
    """Process a single document and display results."""
    
    # Initialize pipeline
    print("Initializing OCR Pipeline...")
    pipeline = OCRPipeline('config.yaml')
    
    # Example: Process an invoice
    image_path = 'path/to/your/invoice.jpg'  # Replace with actual path
    document_type = 'invoice'  # or 'id_document'
    
    print(f"\nProcessing document: {image_path}")
    print(f"Document type: {document_type}")
    print("-" * 60)
    
    # Process document
    result = pipeline.process_document(image_path, document_type)
    
    # Display results
    print(f"\n{'='*60}")
    print(f"PROCESSING RESULTS")
    print(f"{'='*60}")
    
    print(f"\nDecision: {result.decision.upper()}")
    print(f"Final Confidence Score: {result.confidence.final_score:.3f}")
    print(f"Processing Time: {result.processing_time:.2f}s")
    
    print(f"\n{'='*60}")
    print(f"CONFIDENCE BREAKDOWN")
    print(f"{'='*60}")
    print(f"Image Quality:     {result.confidence.image_quality_score:.3f}")
    print(f"OCR Confidence:    {result.confidence.ocr_confidence_score:.3f}")
    print(f"Semantic Score:    {result.confidence.semantic_score:.3f}")
    print(f"Layout Score:      {result.confidence.layout_score:.3f}")
    print(f"Consistency Score: {result.confidence.consistency_score:.3f}")
    
    print(f"\n{'='*60}")
    print(f"EXTRACTED FIELDS")
    print(f"{'='*60}")
    for field_name, field_value in result.extracted_fields.items():
        print(f"{field_name}: {field_value}")
    
    print(f"\n{'='*60}")
    print(f"DECISION REASONS")
    print(f"{'='*60}")
    for reason in result.decision_result.reasons:
        print(f"• {reason}")
    
    print(f"\n{'='*60}")
    print(f"QUALITY METRICS")
    print(f"{'='*60}")
    print(f"Blur Score:       {result.quality_metrics.blur_score:.2f}")
    print(f"Brightness:       {result.quality_metrics.brightness_score:.2f}")
    print(f"Contrast:         {result.quality_metrics.contrast_score:.2f}")
    print(f"Edge Density:     {result.quality_metrics.edge_density:.4f}")
    print(f"Quality Passed:   {result.quality_metrics.passed}")
    
    print(f"\n{'='*60}")
    print(f"OCR STATISTICS")
    print(f"{'='*60}")
    print(f"Total Words:           {result.ocr_result.total_words}")
    print(f"Mean Confidence:       {result.ocr_result.mean_confidence:.2f}")
    print(f"Low Confidence Words:  {result.ocr_result.low_confidence_words}")
    print(f"Numeric Words:         {result.ocr_result.numeric_words}")
    
    # Save full result to JSON
    output_path = 'result.json'
    with open(output_path, 'w') as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
    
    print(f"\n\nFull result saved to: {output_path}")


if __name__ == '__main__':
    main()
