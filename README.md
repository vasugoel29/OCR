# OCR Pipeline

A high-performance, self-hosted OCR system designed to extract structured data from Indian Identity Documents (Aadhaar, PAN, Vehicle RC). It utilizes a multi-layered approach involving image preprocessing, document detection, dual-pass OCR, and a 10-component validation system.

## Quick Start

### Installation

```bash
# Install Python dependencies
pip install -r requirements.txt
```

### Running the API Server

The system provides a FastAPI-based server for processing images via URL:

```bash
python3 api_server.py
```
The server will be available at `http://localhost:8000`.

**Endpoints:**
- `POST /ocr/process_url`: Standard endpoint. Automatically detects type unless `document_type` is provided in the JSON body.
- `POST /ocr/process_url/{doc_type}`: Explicitly sets the document type (e.g., `/ocr/process_url/aadhaar`).

### Command Line Usage

#### 1. Auto-Detect Document Type (Recommended)
The pipeline automatically classifies the document type (Aadhaar, PAN, or RC):

```bash
python3 -m src.pipeline document.jpg
```

#### 2. Process Specific Document Types
You can force a specific extractor if needed:

```bash
# Aadhaar
python3 -m src.pipeline aadhaar.jpg --type aadhaar

# PAN Card
python3 -m src.pipeline pan.jpg --type pan

# Vehicle RC
python3 -m src.pipeline rc.jpg --type vehicle_rc
```

#### 3. Debugging Features
View raw OCR text or save the final extraction results:

```bash
# Show extracted text in console
python3 -m src.pipeline document.jpg --show-text

# Save JSON results
python3 -m src.pipeline document.jpg --output result.json
```

## Python API

```python
from src.pipeline import OCRPipeline

# Initialize pipeline
pipeline = OCRPipeline()

# Process with auto-detection
result = pipeline.process_document('document.jpg', document_type='auto')

# Access results
print(f"Decision: {result.decision}")
print(f"Confidence: {result.confidence.final_score:.3f}")
print(f"Extracted Fields: {result.extracted_fields}")
```

## Features

### 🧠 Core Orchestration
- **Dual-Pass OCR**: Intelligently combines standard and enhanced preprocessing passes to maximize extraction accuracy.
- **Multilingual Support**: Supports English and Hindi/Devanagari text extraction with specialized numeral normalization.
- **Document Classification**: Automatically identifies document types based on keyword frequency and spatial patterns.

### 📉 10-Component Scoring Model
The system explains its decisions via a weighted confidence score [0-1] based on:
1.  **Image Quality**: Blur, brightness, and contrast checks.
2.  **OCR Confidence**: Character-level metadata from PaddleOCR.
3.  **Regex Pattern Match**: Verification against strict document formats.
4.  **Fuzzy Matching**: Anchor word detection (e.g., "Father's Name").
5.  **Layout Analysis**: Verification of physical field locations.
6.  **Key-Value Pair Proximity**: Spatial relationship between keys and values.
7.  **Cross-Field Consistency**: Logical checks between related fields.
8.  **Schema Compliance**: Ensuring all mandatory fields are found.
9.  **Token Distribution**: Analysis of numeric vs. alphabetic ratios.
10. **Spatial Compactness**: Prevents cross-region mixing of fields.

### 🖼️ Image Intelligence
- **Quality Gate**: Rejects blurry or poorly lit images before heavy processing.
- **Auto-Deskewing**: Hough line transform to correct rotated documents.
- **ID Enhancer**: Specialized filters to sharpen small fonts and improve contrast.

## Project Structure

```
OCR-Paddle/
├── api_server.py            # FastAPI service entry point
├── config.yaml              # Thresholds and model configurations
├── src/                    
│   ├── pipeline.py          # Main orchestration logic
│   ├── classification.py    # Document type detection
│   ├── documents/           # Extractors (aadhaar.py, pan.py, vehicle_rc.py)
│   ├── ocr/                 # PaddleOCR wrapper and engine
│   ├── preprocessing/       # Image correction and enhancement
│   ├── quality/             # Assessment of blur/brightness
│   ├── scoring/             # Weighted confidence & decision logic
│   ├── validation/          # Normalization and business rules
│   └── utils.py             # Common image and text utilities
└── tests/                   # Verification suite
```

## License
MIT License
