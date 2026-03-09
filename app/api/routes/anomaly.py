"""
SmartCertify ML — Anomaly Detection API Route
POST /api/ml/anomaly
"""

import time
import logging
from typing import Optional
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.middleware.auth import verify_api_key
from app.models.anomaly.isolation_forest import detect_anomaly
from app.utils.monitoring import log_prediction

logger = logging.getLogger(__name__)
router = APIRouter()


class AnomalyInput(BaseModel):
    issuer_reputation_score: Optional[float] = Field(None, ge=0, le=1)
    certificate_age_days: Optional[int] = None
    metadata_completeness_score: Optional[float] = Field(None, ge=0, le=1)
    ocr_confidence_score: Optional[float] = Field(None, ge=0, le=1)
    template_match_score: Optional[float] = Field(None, ge=0, le=1)
    domain_verification_status: Optional[int] = Field(None, ge=0, le=1)
    previous_verification_count: Optional[int] = None
    time_since_last_verification_days: Optional[float] = None


@router.post("/anomaly")
async def detect_certificate_anomaly(
    data: AnomalyInput,
    api_key: str = Depends(verify_api_key),
):
    """Detect anomalous certificate patterns."""
    start_time = time.time()

    cert_data = data.model_dump(exclude_none=False)
    # Replace None values with 0 for the model
    for k, v in cert_data.items():
        if v is None:
            cert_data[k] = 0

    result = detect_anomaly(cert_data)

    latency_ms = (time.time() - start_time) * 1000

    log_prediction(
        endpoint="/api/ml/anomaly",
        input_data=cert_data,
        prediction=result,
        confidence=abs(result.get("anomaly_score", 0)),
        latency_ms=latency_ms,
    )

    result["latency_ms"] = round(latency_ms, 1)
    return result
