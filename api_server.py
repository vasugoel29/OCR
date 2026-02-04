import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import shutil
import tempfile
import os
import sys
import logging
from typing import Dict, Any, Optional

# Ensure src is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import Pipeline
from src.pipeline import OCRPipeline

# Config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_server")

app = FastAPI(title="OCR Pipeline API", description="Extract data from Indian Identity Documents (Aadhaar, PAN, Vehicle RC)")

# Global pipeline instance
pipeline = None

@app.on_event("startup")
async def startup_event():
    global pipeline
    logger.info("Initializing OCR Pipeline...")
    # Initialize pipeline once on startup
    pipeline = OCRPipeline()
    logger.info("OCR Pipeline initialized.")

class OCRRequest(BaseModel):
    image_url: str

class OCRResponse(BaseModel):
    status: str
    document_type: str
    decision: str
    confidence_score: float
    reason: str
    extracted_fields: Dict[str, Any]
    processing_time: float

@app.post("/ocr/process_url", response_model=OCRResponse)
async def process_url(request: OCRRequest):
    """
    Process an image from a URL.
    Returns extracted fields, confidence score, and decision reason.
    """
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    tmp_path = None
    try:
        # 1. Download image
        logger.info(f"Fetching image from: {request.image_url}")
        response = requests.get(request.image_url, stream=True, timeout=15)
        response.raise_for_status()
        
        # Check content type (simple check)
        content_type = response.headers.get('content-type', '').lower()
        if not any(t in content_type for t in ['image', 'octet-stream']):
            # octet-stream might be an image, so we'll be lenient but warn
            logger.warning(f"Suspicious Content-Type: {content_type}")
        
        # Determine suffix
        suffix = ".jpg" 
        if "png" in content_type: suffix = ".png"
        elif "jpeg" in content_type: suffix = ".jpg"
        elif "webp" in content_type: suffix = ".webp"
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            shutil.copyfileobj(response.raw, tmp_file)
            tmp_path = tmp_file.name
        
        logger.info(f"Image saved to {tmp_path}")

        # 2. Process
        logger.info("Running processing pipeline...")
        result = pipeline.process_document(tmp_path, document_type='auto')
        
        # 3. Extract Reason
        reason = "-"
        if hasattr(result, 'decision_result') and result.decision_result:
            if result.decision_result.reasons:
                reason = result.decision_result.reasons[0]
            
        # 4. Response
        response_data = OCRResponse(
            status="success",
            document_type=result.document_type,
            decision=result.decision,
            confidence_score=round(result.confidence.final_score, 3),
            reason=reason,
            extracted_fields=result.extracted_fields,
            processing_time=round(result.processing_time, 2)
        )
        
        logger.info(f"Processing complete: {result.decision} (Score: {response_data.confidence_score})")
        return response_data
        
    except requests.RequestException as e:
        logger.error(f"Network error fetching image: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to fetch image: {str(e)}")
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Cleanup
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                logger.debug(f"Removed temp file {tmp_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temp file: {e}")

if __name__ == "__main__":
    # Run server
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)
