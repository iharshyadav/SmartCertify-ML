"""
SmartCertify ML — Image Analysis API Route
POST /api/ml/analyze-image
"""

import time
import logging
from typing import Optional
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.middleware.auth import verify_api_key
from app.models.image_analysis.cnn_model import analyze_image
from app.utils.monitoring import log_prediction

logger = logging.getLogger(__name__)
router = APIRouter()


class ImageAnalysisInput(BaseModel):
    image_base64: str
    certificate_id: Optional[str] = "unknown"


@router.post("/analyze-image")
async def analyze_certificate_image(
    data: ImageAnalysisInput,
    api_key: str = Depends(verify_api_key),
):
    """Analyze a certificate image for tampering."""
    start_time = time.time()

    result = analyze_image(
        image_input=data.image_base64,
        certificate_id=data.certificate_id,
    )

    latency_ms = (time.time() - start_time) * 1000

    log_prediction(
        endpoint="/api/ml/analyze-image",
        input_data={"certificate_id": data.certificate_id},
        prediction=result,
        confidence=result.get("confidence", 0),
        latency_ms=latency_ms,
    )

    result["latency_ms"] = round(latency_ms, 1)
    return result
