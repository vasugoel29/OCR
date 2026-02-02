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

### Basic Usage

#### Process Single Document

```bash
# Process ID Document (Aadhaar, etc.)
python3 -m src.pipeline document.jpg --type id_document

# Process Invoice
python3 -m src.pipeline invoice.jpg --type invoice

# Save output to JSON
python3 -m src.pipeline document.jpg --type id_document --output result.json
```

### Python API

```python
from src.pipeline import OCRPipeline

# Initialize pipeline
pipeline = OCRPipeline('config.yaml')

# Process invoice
result = pipeline.process_document('invoice.jpg', document_type='invoice')

# Process ID document (Dual-Pass OCR)
result_id = pipeline.process_document('aadhaar.jpg', document_type='id_document')

# Get results
print(f"Text: {result_id.ocr_result.full_text}")
print(f"Fields: {result_id.extracted_fields}")
print(f"Decision: {result_id.decision}")
```

## Features

- **Document Support**:
  - Invoices
  - **ID Documents (Aadhaar, etc.)**: Features specialized **Dual-Pass OCR** (Original + Enhanced) for maximum accuracy on numbers and text.
- **Robustness**: Deskewing and advanced preprocessing for real-world images.
- **Validation**: Image quality, OCR confidence, semantic validation, layout validation, consistency checks.
- **Self-Hosted**: No external APIs, fully deterministic.

## Project Structure

```
OCR/
├── config.yaml              # Configuration
├── requirements.txt         # Dependencies
├── src/                    # Core modules
│   ├── pipeline.py         # Main orchestrator
│   ├── quality/            # Image quality assessment
│   ├── preprocessing/      # Image corrections & enhancements
│   ├── ocr/               # Tesseract integration
│   ├── validation/        # Semantic, layout, consistency
│   ├── scoring/           # Confidence & decision
│   └── documents/         # Document processors (Aadhaar, Invoice)
├── examples/              # Example scripts
└── tests/                 # Test suite
```

## Configuration

Edit `config.yaml` to adjust thresholds:

```yaml
decision:
  accept_threshold: 0.85    # Auto-accept if score ≥ 0.85
  review_threshold: 0.60    # Manual review if 0.60 ≤ score < 0.85

ocr:
  language: 'eng+hin'       # Support English and Hindi
```

## Testing

```bash
# Run tests
pytest tests/ -v
```

## Documentation

- **[README.md](README.md)** - This file
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Technical details
- **[HINDI_SUPPORT.md](HINDI_SUPPORT.md)** - Hindi OCR setup guide

## License

MIT License
