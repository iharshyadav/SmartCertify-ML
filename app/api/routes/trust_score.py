"""
SmartCertify ML — Trust Score API Route
POST /api/ml/trust-score
"""

import time
import logging
from typing import Optional
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.middleware.auth import verify_api_key
from app.models.trust_score.regression_model import predict_trust_score
from app.models.trust_score.time_series import get_verification_trends
from app.utils.monitoring import log_prediction

logger = logging.getLogger(__name__)
router = APIRouter()


class TrustScoreInput(BaseModel):
    issuer_id: Optional[str] = ""
    total_certificates_issued: Optional[int] = 100
    fraud_rate_historical: Optional[float] = 0.01
    avg_metadata_completeness: Optional[float] = 0.8
    domain_age_days: Optional[int] = 365
    verification_success_rate: Optional[float] = 0.9
    response_time_avg: Optional[float] = 50.0


@router.post("/trust-score")
async def get_trust_score(
    data: TrustScoreInput,
    api_key: str = Depends(verify_api_key),
):
    """Predict issuer trust score."""
    start_time = time.time()

    features = data.model_dump()
    issuer_id = features.pop("issuer_id", "")

    result = predict_trust_score(features)

    # Add trend data
    try:
        trends = get_verification_trends()
        result["trend"] = trends.get("forecast", {}).get("trend", "stable")
    except Exception:
        result["trend"] = "unavailable"

    latency_ms = (time.time() - start_time) * 1000

    log_prediction(
        endpoint="/api/ml/trust-score",
        input_data=features,
        prediction=result,
        confidence=result.get("confidence", 0),
        latency_ms=latency_ms,
    )

    result["issuer_id"] = issuer_id
    result["latency_ms"] = round(latency_ms, 1)
    return result
