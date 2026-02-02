"""Example: Process multiple documents in batch."""

import sys
from pathlib import Path
import json
from collections import Counter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pipeline import OCRPipeline


def main():
    """Process multiple documents in batch and generate summary report."""
    
    # Initialize pipeline
    print("Initializing OCR Pipeline...")
    pipeline = OCRPipeline('config.yaml')
    
    # List of documents to process
    image_paths = [
        'path/to/invoice1.jpg',
        'path/to/invoice2.jpg',
        'path/to/invoice3.jpg',
        # Add more paths...
    ]
    
    document_type = 'invoice'  # or 'id_document'
    
    print(f"\nProcessing batch of {len(image_paths)} documents...")
    print(f"Document type: {document_type}")
    print("-" * 60)
    
    # Process batch
    results = pipeline.process_batch(image_paths, document_type, max_workers=4)
    
    # Generate summary statistics
    decisions = Counter(r.decision for r in results)
    total_time = sum(r.processing_time for r in results)
    avg_time = total_time / len(results) if results else 0
    avg_confidence = sum(r.confidence.final_score for r in results) / len(results) if results else 0
    
    # Display summary
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total Documents:     {len(results)}")
    print(f"Total Time:          {total_time:.2f}s")
    print(f"Average Time:        {avg_time:.2f}s per document")
    print(f"Average Confidence:  {avg_confidence:.3f}")
    
    print(f"\n{'='*60}")
    print(f"DECISION BREAKDOWN")
    print(f"{'='*60}")
    print(f"Accepted:  {decisions.get('accept', 0):3d} ({decisions.get('accept', 0)/len(results)*100:.1f}%)")
    print(f"Review:    {decisions.get('review', 0):3d} ({decisions.get('review', 0)/len(results)*100:.1f}%)")
    print(f"Rejected:  {decisions.get('reject', 0):3d} ({decisions.get('reject', 0)/len(results)*100:.1f}%)")
    print(f"Errors:    {decisions.get('error', 0):3d} ({decisions.get('error', 0)/len(results)*100:.1f}%)")
    
    # Display individual results
    print(f"\n{'='*60}")
    print(f"INDIVIDUAL RESULTS")
    print(f"{'='*60}")
    print(f"{'Document':<40} {'Decision':<10} {'Score':<8} {'Time':<8}")
    print("-" * 60)
    
    for result in results:
        doc_name = Path(result.document_path).name
        print(f"{doc_name:<40} {result.decision.upper():<10} {result.confidence.final_score:.3f}    {result.processing_time:.2f}s")
    
    # Separate documents by decision
    accepted = [r for r in results if r.decision == 'accept']
    review = [r for r in results if r.decision == 'review']
    rejected = [r for r in results if r.decision == 'reject']
    errors = [r for r in results if r.decision == 'error']
    
    # Save results to separate files
    output_dir = Path('batch_results')
    output_dir.mkdir(exist_ok=True)
    
    # Save accepted
    if accepted:
        with open(output_dir / 'accepted.json', 'w') as f:
            json.dump([r.to_dict() for r in accepted], f, indent=2, default=str)
        print(f"\nAccepted documents saved to: {output_dir / 'accepted.json'}")
    
    # Save review
    if review:
        with open(output_dir / 'review.json', 'w') as f:
            json.dump([r.to_dict() for r in review], f, indent=2, default=str)
        print(f"Review documents saved to: {output_dir / 'review.json'}")
    
    # Save rejected
    if rejected:
        with open(output_dir / 'rejected.json', 'w') as f:
            json.dump([r.to_dict() for r in rejected], f, indent=2, default=str)
        print(f"Rejected documents saved to: {output_dir / 'rejected.json'}")
    
    # Save errors
    if errors:
        with open(output_dir / 'errors.json', 'w') as f:
            json.dump([r.to_dict() for r in errors], f, indent=2, default=str)
        print(f"Error documents saved to: {output_dir / 'errors.json'}")
    
    # Save summary report
    summary = {
        'total_documents': len(results),
        'total_time': total_time,
        'average_time': avg_time,
        'average_confidence': avg_confidence,
        'decisions': dict(decisions),
        'accepted_count': len(accepted),
        'review_count': len(review),
        'rejected_count': len(rejected),
        'error_count': len(errors)
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary report saved to: {output_dir / 'summary.json'}")


if __name__ == '__main__':
    main()
