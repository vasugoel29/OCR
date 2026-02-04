# Project Structure

This document describes the refactored directory structure of the OCR Pipeline project.

## Directory Layout

```
OCR/
├── ocr_pipeline/              # Main package
│   ├── __init__.py            # Package initialization (includes imghdr compatibility)
│   ├── api/                   # API layer
│   │   ├── __init__.py
│   │   ├── models.py          # Pydantic request/response models
│   │   └── server.py           # FastAPI application
│   ├── core/                  # Core pipeline logic
│   │   ├── __init__.py
│   │   ├── pipeline.py         # Main OCRPipeline orchestrator
│   │   └── classification.py   # Document type classification
│   ├── compat/                # Compatibility shims
│   │   ├── __init__.py
│   │   └── imghdr.py          # Python 3.13+ compatibility for paddleocr
│   ├── documents/             # Document-specific extractors
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── aadhaar.py
│   │   ├── pan.py
│   │   └── vehicle_rc.py
│   ├── ocr/                   # OCR engine wrapper
│   │   ├── __init__.py
│   │   ├── engine.py
│   │   └── models.py
│   ├── preprocessing/         # Image preprocessing
│   │   ├── __init__.py
│   │   ├── corrections.py
│   │   ├── id_enhancer.py
│   │   └── pipeline.py
│   ├── quality/               # Image quality assessment
│   │   ├── __init__.py
│   │   └── image_quality.py
│   ├── scoring/               # Confidence scoring and decision
│   │   ├── __init__.py
│   │   ├── confidence.py
│   │   └── decision.py
│   ├── segmentation/          # Document segmentation
│   │   ├── __init__.py
│   │   ├── document_detector.py
│   │   ├── region.py
│   │   ├── segmentation_pipeline.py
│   │   └── text_clustering.py
│   ├── validation/            # Validation modules
│   │   ├── __init__.py
│   │   ├── anchors.py
│   │   ├── business_rules.py
│   │   ├── distribution.py
│   │   ├── key_value.py
│   │   ├── normalization.py
│   │   └── spatial_validator.py
│   └── utils.py               # Utility functions
├── tests/                     # Test suite
│   ├── __init__.py
│   ├── images/               # Test images
│   ├── test_quality.py
│   ├── test_segmentation.py
│   └── test_validation.py
├── config.yaml                # Configuration file
├── conftest.py                # Pytest configuration
├── run.py                     # Entry point script
├── setup.py                   # Package setup
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation
```

## Key Changes

### 1. Package Structure
- **Before**: `src/` directory with flat structure
- **After**: `ocr_pipeline/` proper Python package with organized modules

### 2. API Separation
- **Before**: `api_server.py` at root level
- **After**: `ocr_pipeline/api/` with separated models and server

### 3. Core Logic Organization
- **Before**: `pipeline.py` and `classification.py` in `src/`
- **After**: `ocr_pipeline/core/` for core orchestration logic

### 4. Compatibility Handling
- **Before**: `imghdr.py` at root level
- **After**: `ocr_pipeline/compat/` with automatic Python 3.13+ compatibility

### 5. Entry Point
- **Before**: `python api_server.py`
- **After**: `python run.py` or `ocr-pipeline-api` command

## Import Patterns

### Before
```python
from src.pipeline import OCRPipeline
from src.quality import ImageQualityAssessor
```

### After
```python
from ocr_pipeline import OCRPipeline
from ocr_pipeline.quality import ImageQualityAssessor
from ocr_pipeline.core.pipeline import OCRPipeline
from ocr_pipeline.api.models import OCRRequest, OCRResponse
```

## Running the Application

### Development
```bash
python run.py
```

### Production (after installation)
```bash
ocr-pipeline-api
```

### API Documentation
Once running, access:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Benefits

1. **Better Organization**: Clear separation of concerns (API, core, compatibility)
2. **Proper Package**: Can be installed via pip
3. **Easier Testing**: Tests can import from package directly
4. **Maintainability**: Logical grouping makes code easier to navigate
5. **Scalability**: Easy to add new modules without cluttering root directory
