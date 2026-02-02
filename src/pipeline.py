"""Main OCR pipeline orchestrator.

Coordinates all stages: quality gate → preprocessing → OCR → validation → scoring → decision.
"""

import logging
from pathlib import Path
from typing import Dict, Union, Optional, List
from dataclasses import dataclass, field
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from .utils import load_config, setup_logging, load_image, clean_text
from .quality import ImageQualityAssessor, QualityMetrics
from .preprocessing import PreprocessingPipeline
from .preprocessing.id_enhancer import IDDocumentEnhancer
from .ocr import TesseractEngine, OCRResult
from .documents import InvoiceProcessor, IDDocumentProcessor, BaseDocumentProcessor
from .documents.aadhaar import AadhaarExtractor
from .scoring import ConfidenceScorer, DecisionEngine, DocumentConfidence, DecisionResult, Decision


@dataclass
class PipelineResult:
    """Complete pipeline result for a document."""
    document_path: str
    document_type: str
    decision: str
    confidence: DocumentConfidence
    decision_result: DecisionResult
    extracted_fields: Dict
    quality_metrics: QualityMetrics
    ocr_result: OCRResult
    processing_time: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'document_path': self.document_path,
            'document_type': self.document_type,
            'decision': self.decision,
            'confidence_scores': self.confidence.to_dict(),
            'decision_details': self.decision_result.to_dict(),
            'extracted_fields': self.extracted_fields,
            'quality_metrics': {
                'blur_score': self.quality_metrics.blur_score,
                'brightness_score': self.quality_metrics.brightness_score,
                'resolution_score': self.quality_metrics.resolution_score,
                'contrast_score': self.quality_metrics.contrast_score,
                'edge_density': self.quality_metrics.edge_density,
                'composite_score': self.quality_metrics.composite_score,
                'passed': self.quality_metrics.passed
            },
            'ocr_stats': {
                'total_words': self.ocr_result.total_words,
                'mean_confidence': self.ocr_result.mean_confidence,
                'low_confidence_words': self.ocr_result.low_confidence_words,
                'numeric_words': self.ocr_result.numeric_words
            },
            'processing_time': self.processing_time,
            'error': self.error
        }


class OCRPipeline:
    """Main OCR pipeline for document processing."""
    
    def __init__(self, config_path: Union[str, Path] = "config.yaml"):
        """Initialize OCR pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Setup logging
        self.logger = setup_logging(self.config.get('logging', {}))
        self.logger.info("Initializing OCR Pipeline")
        
        # Initialize components
        self.quality_assessor = ImageQualityAssessor(self.config.get('quality', {}))
        self.preprocessing_pipeline = PreprocessingPipeline(self.config.get('preprocessing', {}))
        self.ocr_engine = TesseractEngine(self.config.get('ocr', {}))
        self.confidence_scorer = ConfidenceScorer(self.config.get('scoring', {}))
        self.decision_engine = DecisionEngine(self.config.get('decision', {}))
        
        # Initialize document processors
        self.invoice_processor = InvoiceProcessor(self.config)
        self.id_processor = IDDocumentProcessor(self.config)
        
        self.logger.info("OCR Pipeline initialized successfully")
    
    def process_document(self,
                        image_path: Union[str, Path],
                        document_type: str = 'invoice',
                        save_intermediates: bool = False) -> PipelineResult:
        """Process a single document through the complete pipeline.
        
        Args:
            image_path: Path to document image
            document_type: Type of document ('invoice' or 'id_document')
            save_intermediates: Whether to save intermediate processing results
            
        Returns:
            PipelineResult object
        """
        import time
        start_time = time.time()
        
        self.logger.info(f"Processing document: {image_path} (type: {document_type})")
        
        try:
            # Load image
            image = load_image(image_path)
            image_shape = image.shape[:2]
            
            # Stage 1: Image Quality Gate
            self.logger.debug("Stage 1: Image Quality Assessment")
            quality_metrics = self.quality_assessor.assess(image)
            
            if not quality_metrics.passed:
                self.logger.warning(f"Image failed quality gate: {quality_metrics.failure_reasons}")
            
            # Stage 2: Preprocessing & Correction
            self.logger.debug("Stage 2: Image Preprocessing")
            preprocessing_result = self.preprocessing_pipeline.process(image, save_intermediates)
            processed_image = preprocessing_result['processed_image']
            
            # Stage 3: OCR
            self.logger.debug("Stage 3: OCR Extraction")
            
            if document_type == 'id_document':
                # Dual-pass OCR for ID documents
                self.logger.info("Using dual-pass OCR for ID document")
                
                # Pass 1: Standard raw image (but deskewed!)
                # Use robust generic skew corrector (Hough lines) instead of simple minAreaRect
                if hasattr(self.preprocessing_pipeline, 'corrector'):
                    deskewed_image = self.preprocessing_pipeline.corrector.correct_skew(image)
                else:
                    # Fallback if pipeline not initialized with corrector (unlikely)
                    deskewed_image = image
                
                ocr_result = self.ocr_engine.extract_text(deskewed_image)
                
                # Pass 2: Enhanced image
                # Pass deskewed image to enhancer for better results
                id_enhancer = IDDocumentEnhancer()
                enhanced_image = id_enhancer.enhance_for_ocr(deskewed_image)
                ocr_result_enh = self.ocr_engine.extract_text(enhanced_image)
                
                # Use result with more words as primary for confidence stats
                if ocr_result_enh.total_words > ocr_result.total_words:
                    primary_ocr_result = ocr_result_enh
                else:
                    primary_ocr_result = ocr_result
            else:
                ocr_result = self.ocr_engine.extract_text(processed_image)
                primary_ocr_result = ocr_result
            
            # Check if text was detected
            text_detected = primary_ocr_result.total_words > 0
            if not text_detected:
                self.logger.warning("No text detected in document")
            
            # Calculate OCR confidence score
            ocr_confidence_score = self.ocr_engine.calculate_ocr_confidence_score(primary_ocr_result)
            
            # Stage 4: Document-Specific Processing
            self.logger.debug("Stage 4: Document-Specific Processing")
            
            if document_type == 'id_document':
                # Specialized Aadhaar/ID extraction
                aadhaar_extractor = AadhaarExtractor()
                
                # Extract from standard pass first
                extracted_fields = aadhaar_extractor.extract_fields(ocr_result)
                
                # If Aadhaar number missing, try enhanced pass
                if 'aadhaar_number' not in extracted_fields:
                    fields_enh = aadhaar_extractor.extract_fields(ocr_result_enh)
                    if 'aadhaar_number' in fields_enh:
                        extracted_fields['aadhaar_number'] = fields_enh['aadhaar_number']
                        # Ensure 'id_number' alias is set for compatibility
                        extracted_fields['id_number'] = fields_enh['aadhaar_number']
                
                # Merge other fields from enhanced pass if missing in standard
                # (Name, DOB, Gender often better in enhanced)
                fields_enh = aadhaar_extractor.extract_fields(ocr_result_enh)
                for key in ['name', 'date_of_birth', 'gender', 'address']:
                    if key not in extracted_fields and key in fields_enh:
                        extracted_fields[key] = fields_enh[key]
                
                # Use ID processor for subsequent validation steps
                processor = self.id_processor
                
                # Update main ocr_result to be the primary one for reporting
                ocr_result = primary_ocr_result
            else:
                processor = self._get_processor(document_type)
                
                # Extract fields
                extracted_fields = processor.extract_fields(ocr_result)
            
            # Check if mandatory fields are present
            required_fields = processor.get_required_fields()
            mandatory_fields_present = all(field in extracted_fields for field in required_fields)
            
            # Validate fields (semantic)
            semantic_validation = processor.validate_fields(extracted_fields)
            semantic_score = semantic_validation['semantic_score']
            
            # Validate layout
            layout_validation = processor.validate_layout(ocr_result, image_shape)
            layout_score = layout_validation['layout_score']
            
            # Check consistency
            consistency_validation = processor.check_consistency(extracted_fields)
            consistency_score = consistency_validation['consistency_score']
            
            # Stage 5: Multi-Stage Confidence Scoring
            self.logger.debug("Stage 5: Confidence Scoring")
            document_confidence = self.confidence_scorer.calculate_document_confidence(
                image_quality_score=quality_metrics.composite_score,
                ocr_confidence_score=ocr_confidence_score,
                semantic_score=semantic_score,
                layout_score=layout_score,
                consistency_score=consistency_score
            )
            
            # Calculate non-alphanumeric ratio
            non_alphanumeric_ratio = self._calculate_non_alphanumeric_ratio(ocr_result.full_text)
            
            # Stage 6: Decision
            self.logger.debug("Stage 6: Decision Making")
            decision_result = self.decision_engine.make_decision(
                document_confidence=document_confidence,
                quality_passed=quality_metrics.passed,
                text_detected=text_detected,
                mandatory_fields_present=mandatory_fields_present,
                non_alphanumeric_ratio=non_alphanumeric_ratio
            )
            
            processing_time = time.time() - start_time
            
            self.logger.info(
                f"Document processed: {decision_result.decision.value} "
                f"(score: {document_confidence.final_score:.3f}, time: {processing_time:.2f}s)"
            )
            
            return PipelineResult(
                document_path=str(image_path),
                document_type=document_type,
                decision=decision_result.decision.value,
                confidence=document_confidence,
                decision_result=decision_result,
                extracted_fields=extracted_fields,
                quality_metrics=quality_metrics,
                ocr_result=ocr_result,
                processing_time=processing_time
            )
        
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Error processing document {image_path}: {str(e)}", exc_info=True)
            
            # Return error result
            return PipelineResult(
                document_path=str(image_path),
                document_type=document_type,
                decision='error',
                confidence=DocumentConfidence(0, 0, 0, 0, 0, 0),
                decision_result=DecisionResult(
                    decision=Decision.REJECT,
                    confidence_score=0.0,
                    reasons=[f"Processing error: {str(e)}"],
                    hard_rejection=True
                ),
                extracted_fields={},
                quality_metrics=QualityMetrics(0, 0, 0, 0, 0, 0, False, [str(e)]),
                ocr_result=OCRResult('', 0.0),
                processing_time=processing_time,
                error=str(e)
            )
    
    def process_batch(self,
                     image_paths: List[Union[str, Path]],
                     document_type: str = 'invoice',
                     max_workers: Optional[int] = None) -> List[PipelineResult]:
        """Process multiple documents in parallel.
        
        Args:
            image_paths: List of paths to document images
            document_type: Type of documents
            max_workers: Maximum number of parallel workers (default from config)
            
        Returns:
            List of PipelineResult objects
        """
        if max_workers is None:
            max_workers = self.config.get('batch', {}).get('max_workers', 4)
        
        self.logger.info(f"Processing batch of {len(image_paths)} documents with {max_workers} workers")
        
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self.process_document, path, document_type): path
                for path in image_paths
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error processing {path}: {str(e)}")
                    results.append(PipelineResult(
                        document_path=str(path),
                        document_type=document_type,
                        decision='error',
                        confidence=DocumentConfidence(0, 0, 0, 0, 0, 0),
                        decision_result=DecisionResult(
                            decision=Decision.REJECT,
                            confidence_score=0.0,
                            reasons=[f"Batch processing error: {str(e)}"],
                            hard_rejection=True
                        ),
                        extracted_fields={},
                        quality_metrics=QualityMetrics(0, 0, 0, 0, 0, 0, False, [str(e)]),
                        ocr_result=OCRResult('', 0.0),
                        error=str(e)
                    ))
        
        self.logger.info(f"Batch processing complete: {len(results)} documents processed")
        
        return results
    
    def _get_processor(self, document_type: str) -> BaseDocumentProcessor:
        """Get appropriate document processor.
        
        Args:
            document_type: Type of document
            
        Returns:
            Document processor instance
        """
        if document_type == 'invoice':
            return self.invoice_processor
        elif document_type == 'id_document':
            return self.id_processor
        else:
            raise ValueError(f"Unknown document type: {document_type}")
    
    def _calculate_non_alphanumeric_ratio(self, text: str) -> float:
        """Calculate ratio of non-alphanumeric characters.
        
        Args:
            text: Input text
            
        Returns:
            Ratio of non-alphanumeric characters
        """
        if not text:
            return 0.0
        
        alphanumeric_count = sum(c.isalnum() or c.isspace() for c in text)
        total_count = len(text)
        
        return 1.0 - (alphanumeric_count / total_count)


def main():
    """Main entry point for command-line usage."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='OCR Pipeline for Invoice and ID Documents')
    parser.add_argument('image_path', help='Path to document image')
    parser.add_argument('--type', choices=['invoice', 'id_document'], default='invoice',
                       help='Document type')
    parser.add_argument('--config', default='config.yaml', help='Path to configuration file')
    parser.add_argument('--output', help='Output JSON file path')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = OCRPipeline(args.config)
    
    # Process document
    result = pipeline.process_document(args.image_path, args.type)
    
    # Convert to dict
    result_dict = result.to_dict()
    
    # Print result
    print(json.dumps(result_dict, indent=2))
    
    # Save to file if specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result_dict, f, indent=2)
        print(f"\nResult saved to {args.output}")


if __name__ == '__main__':
    main()
