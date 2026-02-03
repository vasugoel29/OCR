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
from .validation.normalization import TokenNormalizer
from .validation.anchors import AnchorValidator
from .validation.distribution import DistributionAnalyzer
from .validation.key_value import KeyValueExtractor
from .ocr import PaddleOCREngine, OCRResult
from .documents import InvoiceProcessor, IDDocumentProcessor, BaseDocumentProcessor
from .documents.aadhaar import AadhaarExtractor
from .scoring import ConfidenceScorer, DecisionEngine, DocumentConfidence, DecisionResult, Decision
from .classification import DocumentClassifier


@dataclass
class PipelineResult:
    """Complete pipeline result for a document."""
    document_path: str
    document_type: str
    decision: str
    confidence: DocumentConfidence
    decision_result: DecisionResult
    extracted_fields: Dict
    quality_metrics: Dict # Changed to Dict
    ocr_stats: Dict # Changed from ocr_result to ocr_stats dict
    full_text: str = "" # Added
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
            'quality_metrics': self.quality_metrics, # Already a dict
            'ocr_stats': self.ocr_stats,
            'full_text': self.full_text,
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
        self.ocr_engine = PaddleOCREngine(self.config.get('ocr', {}))
        self.confidence_scorer = ConfidenceScorer(self.config.get('scoring', {}))
        self.decision_engine = DecisionEngine(self.config.get('decision', {}))
        self.classifier = DocumentClassifier()
        
        # Initialize validation components
        self.anchor_validator = AnchorValidator(self.config)
        self.distribution_analyzer = DistributionAnalyzer(self.config)
        self.kv_extractor = KeyValueExtractor(self.config)
        self.token_normalizer = TokenNormalizer()
        
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
            
            # Run standard OCR first (needed for classification and general text)
            ocr_result = self.ocr_engine.extract_text(processed_image)
            
            # Auto-classify if needed
            if document_type == 'auto':
                self.logger.info("Auto-detecting document type...")
                
                # If standard pass failed to find text, it might be a challenging ID document
                # Try running the ID enhancement pass blindly to see if we find anything
                if ocr_result.total_words == 0:
                    self.logger.info("No text in standard pass. Trying ID enhancement for classification...")
                    
                    if hasattr(self.preprocessing_pipeline, 'corrector'):
                        deskewed_image = self.preprocessing_pipeline.corrector.correct_skew(image)
                    else:
                        deskewed_image = image
                        
                    id_enhancer = IDDocumentEnhancer()
                    enhanced_image = id_enhancer.enhance_for_ocr(deskewed_image)
                    ocr_result_enh = self.ocr_engine.extract_text(enhanced_image)
                    
                    if ocr_result_enh.total_words > 0:
                        ocr_result = ocr_result_enh  # Use this for classification
                        self.logger.info(f"Enhancement found {ocr_result.total_words} words. Proceeding with classification.")
                
                if ocr_result.total_words == 0:
                    self.logger.warning("No text detected even after enhancement, defaulting to 'invoice'")
                    document_type = 'invoice'
                else:
                    document_type = self.classifier.classify(ocr_result.full_text)
                    self.logger.info(f"Detected document type: {document_type}")
            
            primary_ocr_result = ocr_result
            
            # ID-Specific Enhanced OCR Pass (Dual Pass)
            # If identified as ID document, run the deskew+enhance pass for better accuracy on specific fields
            if document_type == 'id_document':
                self.logger.info("Running enhance pass for ID document")
                
                # Pass 1 was already done above (OCRResult) on standard processed image
                # Pass 2: Enhanced image
                # Use robust skew correction for the base of enhancement
                if hasattr(self.preprocessing_pipeline, 'corrector'):
                    deskewed_image = self.preprocessing_pipeline.corrector.correct_skew(image)
                else:
                    deskewed_image = image
                
                id_enhancer = IDDocumentEnhancer()
                enhanced_image = id_enhancer.enhance_for_ocr(deskewed_image)
                ocr_result_enh = self.ocr_engine.extract_text(enhanced_image)
                
                # Use the pass with more detected words as primary for confidence stats
                if ocr_result_enh.total_words > ocr_result.total_words:
                    primary_ocr_result = ocr_result_enh
                
                # Note: We keep both results available for the AdhaarExtractor later logic
            else:
                 # Invoice or others - single pass is usually sufficient
                 ocr_result_enh = None
                 primary_ocr_result = ocr_result
            
            # Check if text was detected
            
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
                
                # If Aadhaar number missing, try enhanced pass if available
                if 'aadhaar_number' not in extracted_fields and ocr_result_enh:
                    fields_enh = aadhaar_extractor.extract_fields(ocr_result_enh)
                    if 'aadhaar_number' in fields_enh:
                        extracted_fields['aadhaar_number'] = fields_enh['aadhaar_number']
                        # Ensure 'id_number' alias is set for compatibility
                        extracted_fields['id_number'] = fields_enh['aadhaar_number']
                
                # Merge other fields from enhanced pass if missing in standard
                # (Name, DOB, Gender often better in enhanced)
                if ocr_result_enh:
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
            
            # Stage 5: Advanced Validation Layer
            self.logger.debug("Stage 5: Post-OCR Validation")
            
            # 5.1 Token Normalization (in-place or separate? keeping raw text for now, using norm helper)
            # 5.2 Regex Score (implicit in semantic) -> we'll explicitly calculate one
            # Just approximation based on extracted vs required
            regex_score = len(extracted_fields) / len(processor.get_required_fields()) if processor.get_required_fields() else 1.0
            
            # 5.3 Fuzzy Anchors
            fuzzy_score, anchor_details = self.anchor_validator.validate_anchors(ocr_result.full_text, document_type)
            
            # 5.4 Layout (already done)
            
            # 5.5 KV Proximity
            kv_score = self.kv_extractor.validate_kv_pairs(ocr_result, document_type)
            
            # 5.6 Consistency (already done)
            
            # 5.7 Distribution
            distribution_score, dist_metrics = self.distribution_analyzer.analyze(ocr_result.full_text, document_type)
            
            # 5.8 Schema Completeness
            required_count = len(required_fields)
            present_count = sum(1 for f in required_fields if f in extracted_fields)
            schema_score = present_count / required_count if required_count > 0 else 1.0
            
            # Stage 6: Multi-Stage Confidence Scoring
            self.logger.debug("Stage 6: Confidence Scoring")
            document_confidence = self.confidence_scorer.calculate_document_confidence(
                image_quality_score=quality_metrics.composite_score,
                ocr_confidence_score=ocr_confidence_score,
                regex_score=regex_score,
                fuzzy_score=fuzzy_score,
                layout_score=layout_score,
                kv_score=kv_score,
                consistency_score=consistency_score,
                schema_score=schema_score,
                distribution_score=distribution_score
            )
            
            # Calculate non-alphanumeric ratio
            non_alphanumeric_ratio = self._calculate_non_alphanumeric_ratio(ocr_result.full_text)
            
            # Stage 7: Decision
            self.logger.debug("Stage 7: Decision Making")
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
                quality_metrics=quality_metrics.to_dict(),
                ocr_stats=primary_ocr_result.get_stats(),
                full_text=primary_ocr_result.full_text,
                processing_time=processing_time,
                error=None
            )
        
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Error processing document {image_path}: {str(e)}", exc_info=True)
            
            # Return error result
            # Return error result
            return PipelineResult(
                document_path=str(image_path),
                document_type=document_type,
                decision='error',
                confidence=DocumentConfidence(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                decision_result=DecisionResult(
                    decision=Decision.REJECT,
                    confidence_score=0.0,
                    reasons=[f"Processing error: {str(e)}"],
                    hard_rejection=True
                ),
                extracted_fields={},
                quality_metrics={'error': str(e)}, # Manual dict
                ocr_stats={'error': str(e)}, # Manual dict
                full_text="", # Manual empty
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
                        confidence=DocumentConfidence(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
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
    parser.add_argument('--type', choices=['invoice', 'id_document', 'auto'], default='auto',
                       help='Document type (default: auto)')
    parser.add_argument('--config', default='config.yaml', help='Path to configuration file')
    parser.add_argument('--output', help='Output JSON file path')
    parser.add_argument('--show-text', action='store_true', help='Print full OCR text to stdout')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = OCRPipeline(args.config)
    
    # Process document
    try:
        result = pipeline.process_document(args.image_path, document_type=args.type)
        
        # Print full text if requested
        if args.show_text and hasattr(result, 'full_text') and result.full_text:
            print("\n" + "="*50)
            print("EXTRACTED TEXT")
            print("="*50)
            print(result.full_text)
            print("="*50 + "\n")
            
        print(json.dumps(result.to_dict(), indent=2))
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"\nResult saved to {args.output}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
