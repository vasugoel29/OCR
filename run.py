#!/usr/bin/env python3
"""Entry point for running the OCR Pipeline API server."""

# Import ocr_pipeline first to set up compatibility shims
import ocr_pipeline  # noqa: F401

# Now import and run the API server
from ocr_pipeline.api.server import app
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
