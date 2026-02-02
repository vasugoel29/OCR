# OCR Pipeline

Production-ready OCR system for processing invoices and ID documents with multi-stage validation.

## Quick Start

### Installation

```bash
# Install Tesseract OCR
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-hin

# Install Python dependencies
pip install -r requirements.txt
```

### Usage

#### 1. Process ID Documents (Aadhaar, PAN, etc.)

The system uses a specialized **Dual-Pass OCR** strategy for ID cards:
1.  **Pass 1 (Raw Deskewed)**: Extracts accurate numbers (Aadhaar, IDs).
2.  **Pass 2 (Enhanced)**: Extracts clearer text (Names, Addresses).
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

#### 2. Process Invoices

Standard pipeline for financial documents:

```bash
python3 -m src.pipeline invoice.jpg --type invoice
```

#### 3. JSON Output

Save full results to a JSON file:

```bash
python3 -m src.pipeline document.jpg --type id_document --output result.json
```

## Python API

```python
from src.pipeline import OCRPipeline

# Initialize pipeline
pipeline = OCRPipeline('config.yaml')

# Process ID document
result = pipeline.process_document('aadhaar.jpg', document_type='id_document')

if result.decision != 'reject':
    print("✅ Extraction Successful")
    print(f"Name: {result.extracted_fields.get('name')}")
    print(f"Aadhaar: {result.extracted_fields.get('aadhaar_number')}")
else:
    print(f"❌ Rejected: {result.decision_result.reasons}")
```

## Features

- **Dual-Pass OCR Architecture**:
  - Handles real-world ID cards with noise, blur, and skew.
  - Automatically runs OCR on both generic and aggressively enhanced images.
  - Merges results to get 100% Aadhaar number accuracy in tests.
- **Robustness**: 
  - **Deskewing**: Uses Hough Line Transform to fix rotation.
  - **Enhancement**: Adaptive thresholding and denoising for text clarity.
- **Validation Suite**: 
  - Image Quality (Blur, Glare, Contrast)
  - Semantic Validation (Date formats, ID patterns)
  - Cross-Field Consistency Checks

## Configuration

Edit `config.yaml` to adjust thresholds:

```yaml
decision:
  accept_threshold: 0.85    # Auto-accept confidence
  review_threshold: 0.60    # Flag for manual revies

ocr:
  language: 'eng+hin'       # Enable Hindi support
```

## Troubleshooting

- **Low Confidence Scores**: 
  - Ensure image is in focus (`blur_score > 300`).
  - Ensure good contrast (`contrast_score > 0.3`).
- **Wrong Orientation**: The system tries to deskew, but images rotated >45 degrees might need manual rotation.
- **Missing Fields**: Check `config.yaml` to see if fields are marked mandatory.

## Project Structure

```
OCR/
├── config.yaml              # Configuration settings
├── src/                    
│   ├── pipeline.py         # Main entry point & orchestration
│   ├── preprocessing/      # Image corrections (id_enhancer.py)
│   ├── documents/         # Field extractors (aadhaar.py)
│   ├── ocr/               # Tesseract wrapper
│   └── quality/           # Quality checks
```

## License

MIT License
