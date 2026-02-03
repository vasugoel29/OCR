# OCR Pipeline

Production-ready OCR system for processing invoices and ID documents with multi-stage validation, powered by PaddleOCR.

## Quick Start

### Installation

```bash
# Install Python dependencies
pip install -r requirements.txt
```

### Usage

#### 1. Auto-Detect Document Type (Recommended)

The pipeline automatically detects whether a document is an ID card or invoice:

```bash
python3 -m src.pipeline document.jpg
```

#### 2. Process ID Documents (Aadhaar, PAN, etc.)

The system uses a specialized **Dual-Pass OCR** strategy for ID cards:
1.  **Pass 1 (Standard)**: Initial OCR extraction.
2.  **Pass 2 (Enhanced)**: High-resolution preprocessing with deskewing and adaptive thresholding.
3.  **Merge**: Combines best results from both passes.

```bash
python3 -m src.pipeline aadhaar.jpg --type id_document
```

**Extracted Fields:**
- `aadhaar_number` / `id_number`
- `name`
- `date_of_birth`
- `gender`
- `address` (if legible)
- `vid` (Virtual ID)

#### 3. Process Invoices

```bash
python3 -m src.pipeline invoice.jpg --type invoice
```

#### 4. Debug with Raw Text Output

View the raw OCR text extracted from the document:

```bash
python3 -m src.pipeline document.jpg --show-text
```

#### 5. JSON Output

Save full results to a JSON file:

```bash
python3 -m src.pipeline document.jpg --output result.json
```

## Python API

```python
from src.pipeline import OCRPipeline

# Initialize pipeline
pipeline = OCRPipeline('config.yaml')

# Process with auto-detection
result = pipeline.process_document('document.jpg', document_type='auto')

# Access results
print(f"Decision: {result.decision}")
print(f"Confidence: {result.confidence.final_score:.2f}")
print(f"Extracted Fields: {result.extracted_fields}")

# Convert to dictionary for serialization
result_dict = result.to_dict()
```

## Features

### Core Capabilities
- **PaddleOCR Engine**: High-accuracy OCR with support for English and Hindi
- **Automatic Document Classification**: Detects ID documents vs invoices using keyword and pattern analysis
- **Dual-Pass OCR for IDs**: Combines standard and enhanced preprocessing for optimal field extraction
- **Robust Field Extraction**: Handles OCR noise (merged text, special characters, missing separators)

### Advanced Validation
- **9-Component Confidence Scoring**:
  - Image Quality Score
  - OCR Confidence Score
  - Regex Pattern Match Score
  - Fuzzy Matching Score
  - Layout Analysis Score
  - Key-Value Pair Score
  - Cross-Field Consistency Score
  - Schema Compliance Score
  - Token Distribution Score

- **Multi-Stage Decision Engine**:
  - `accept`: High confidence (≥0.85)
  - `review`: Medium confidence (0.60-0.85) - requires manual review
  - `reject`: Low confidence or failed validation

### Image Preprocessing
- **Quality Gate**: Checks blur, brightness, contrast, and edge density
- **ID Enhancement**: High-resolution upscaling (1600px), denoising, CLAHE contrast enhancement
- **Deskewing**: Hough Line Transform for rotation correction

## Configuration

Edit `config.yaml` to adjust thresholds:

```yaml
decision:
  accept_threshold: 0.85    # Auto-accept confidence
  review_threshold: 0.60    # Flag for manual review

paddle_ocr:
  lang: 'en'                # Language model
  use_angle_cls: true       # Enable rotation detection
  use_gpu: false            # Set to true if GPU available

quality:
  min_blur_score: 100.0     # Minimum sharpness
  min_brightness: 20        # Brightness range
  max_brightness: 240
```

## Troubleshooting

- **Low Confidence Scores**: 
  - Ensure image is in focus (`blur_score > 100`)
  - Ensure good contrast (`contrast_score > 0.2`)
  - Check image resolution (minimum 640x480)
- **Wrong Classification**: Use `--type` to force document type
- **Missing Fields**: Check logs for extraction details, use `--show-text` to inspect raw OCR output

## Project Structure

```
OCR/
├── config.yaml              # Configuration settings
├── src/                    
│   ├── pipeline.py          # Main orchestration
│   ├── classification.py    # Document type detection
│   ├── preprocessing/       # Image enhancement (id_enhancer.py)
│   ├── documents/           # Field extractors (aadhaar.py, invoice.py)
│   ├── ocr/                 # PaddleOCR wrapper
│   ├── quality/             # Image quality assessment
│   ├── scoring/             # Confidence scoring & decision engine
│   └── validation/          # Post-OCR validation modules
└── tests/                   # Unit tests
```

## License

MIT License
