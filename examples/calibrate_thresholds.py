"""Example: Calibrate decision thresholds using labeled dataset."""

import sys
from pathlib import Path
import json
import numpy as np
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pipeline import OCRPipeline
from utils import load_config
import yaml


def load_labeled_dataset(dataset_path):
    """Load labeled dataset.
    
    Expected format:
    {
        "documents": [
            {
                "path": "path/to/image.jpg",
                "type": "invoice",
                "label": "accept"  # or "review" or "reject"
            },
            ...
        ]
    }
    """
    with open(dataset_path, 'r') as f:
        return json.load(f)


def calculate_metrics(predictions, labels):
    """Calculate precision, recall, F1 for each class."""
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("\nClassification Report:")
    print(classification_report(labels, predictions))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(labels, predictions))
    
    return classification_report(labels, predictions, output_dict=True)


def find_optimal_thresholds(results, labels):
    """Find optimal accept and review thresholds.
    
    Uses grid search to maximize F1 score.
    """
    accept_thresholds = np.arange(0.70, 0.95, 0.05)
    review_thresholds = np.arange(0.50, 0.80, 0.05)
    
    best_f1 = 0
    best_accept = 0.85
    best_review = 0.60
    
    print("\nSearching for optimal thresholds...")
    print(f"{'Accept':<8} {'Review':<8} {'F1 Score':<10}")
    print("-" * 30)
    
    for accept_thresh in accept_thresholds:
        for review_thresh in review_thresholds:
            if review_thresh >= accept_thresh:
                continue
            
            # Apply thresholds
            predictions = []
            for result in results:
                score = result.confidence.final_score
                if score >= accept_thresh:
                    predictions.append('accept')
                elif score >= review_thresh:
                    predictions.append('review')
                else:
                    predictions.append('reject')
            
            # Calculate F1
            from sklearn.metrics import f1_score
            f1 = f1_score(labels, predictions, average='weighted')
            
            print(f"{accept_thresh:.2f}     {review_thresh:.2f}     {f1:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_accept = accept_thresh
                best_review = review_thresh
    
    print(f"\nBest thresholds found:")
    print(f"Accept threshold:  {best_accept:.2f}")
    print(f"Review threshold:  {best_review:.2f}")
    print(f"F1 Score:          {best_f1:.4f}")
    
    return best_accept, best_review


def main():
    """Calibrate thresholds using labeled dataset."""
    
    # Load labeled dataset
    dataset_path = 'labeled_dataset.json'  # Create this file with your labeled data
    
    print("Loading labeled dataset...")
    try:
        dataset = load_labeled_dataset(dataset_path)
    except FileNotFoundError:
        print(f"Error: {dataset_path} not found.")
        print("\nPlease create a labeled dataset file with the following format:")
        print(json.dumps({
            "documents": [
                {
                    "path": "path/to/image.jpg",
                    "type": "invoice",
                    "label": "accept"
                }
            ]
        }, indent=2))
        return
    
    documents = dataset['documents']
    print(f"Loaded {len(documents)} labeled documents")
    
    # Initialize pipeline with current config
    print("\nInitializing OCR Pipeline...")
    pipeline = OCRPipeline('config.yaml')
    
    # Process all documents
    print("\nProcessing documents...")
    results = []
    labels = []
    
    for i, doc in enumerate(documents, 1):
        print(f"Processing {i}/{len(documents)}: {doc['path']}")
        result = pipeline.process_document(doc['path'], doc['type'])
        results.append(result)
        labels.append(doc['label'])
    
    # Current performance
    print("\n" + "="*60)
    print("CURRENT PERFORMANCE (with existing thresholds)")
    print("="*60)
    
    current_predictions = [r.decision for r in results]
    current_metrics = calculate_metrics(current_predictions, labels)
    
    # Find optimal thresholds
    print("\n" + "="*60)
    print("THRESHOLD OPTIMIZATION")
    print("="*60)
    
    best_accept, best_review = find_optimal_thresholds(results, labels)
    
    # Update config with new thresholds
    config = load_config('config.yaml')
    config['decision']['accept_threshold'] = float(best_accept)
    config['decision']['review_threshold'] = float(best_review)
    
    # Save updated config
    output_config_path = 'config_calibrated.yaml'
    with open(output_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\n\nCalibrated configuration saved to: {output_config_path}")
    print("\nTo use the calibrated thresholds, either:")
    print(f"1. Replace config.yaml with {output_config_path}")
    print(f"2. Use OCRPipeline('{output_config_path}') in your code")
    
    # Save calibration report
    report = {
        'dataset_size': len(documents),
        'current_thresholds': {
            'accept': pipeline.decision_engine.accept_threshold,
            'review': pipeline.decision_engine.review_threshold
        },
        'current_performance': current_metrics,
        'optimized_thresholds': {
            'accept': best_accept,
            'review': best_review
        }
    }
    
    with open('calibration_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("Calibration report saved to: calibration_report.json")


if __name__ == '__main__':
    main()
