"""
SmartCertify ML — Fraud Detection API Route
POST /api/ml/verify
"""

import time
import logging
from typing import Optional
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.middleware.auth import verify_api_key
from app.models.fraud_detection.predict import predict_fraud
from app.models.fraud_detection.explain import explain_prediction
from app.data.preprocess import preprocess_single
from app.utils.monitoring import log_prediction

logger = logging.getLogger(__name__)
router = APIRouter()


class CertificateInput(BaseModel):
    issuer_name: Optional[str] = "Unknown"
    recipient_name: Optional[str] = "Unknown"
    course_name: Optional[str] = "Unknown"
    issue_date: Optional[str] = None
    expiry_date: Optional[str] = None
    credential_hash: Optional[str] = None
    issuer_reputation_score: Optional[float] = Field(None, ge=0, le=1)
    certificate_age_days: Optional[int] = None
    metadata_completeness_score: Optional[float] = Field(None, ge=0, le=1)
    ocr_confidence_score: Optional[float] = Field(None, ge=0, le=1)
    template_match_score: Optional[float] = Field(None, ge=0, le=1)
    domain_verification_status: Optional[int] = Field(None, ge=0, le=1)
    previous_verification_count: Optional[int] = None
    time_since_last_verification_days: Optional[float] = None


@router.post("/verify")
async def verify_certificate(
    cert: CertificateInput,
    api_key: str = Depends(verify_api_key),
):
    """Verify a certificate for fraud detection."""
    start_time = time.time()

    cert_data = cert.model_dump(exclude_none=False)

    # Run prediction
    result = predict_fraud(cert_data)

    if "error" not in result:
        # Add SHAP explanation
        try:
            X = preprocess_single(cert_data)
            explanation = explain_prediction(X)
            result["shap_explanation"] = explanation.get("top_features", [])
            result["top_3_features"] = [
                {"feature": f["feature"], "impact": f.get("impact_direction", f.get("importance", ""))}
                for f in explanation.get("top_features", [])[:3]
            ]
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            result["shap_explanation"] = []
            result["top_3_features"] = []

    latency_ms = (time.time() - start_time) * 1000

    # Log prediction
    log_prediction(
        endpoint="/api/ml/verify",
        input_data=cert_data,
        prediction=result,
        confidence=result.get("confidence_score", 0),
        latency_ms=latency_ms,
    )

    result["latency_ms"] = round(latency_ms, 1)
    return result
