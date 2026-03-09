"""
SmartCertify ML — Similarity API Route
POST /api/ml/similarity
"""

import time
import logging
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.middleware.auth import verify_api_key
from app.utils.monitoring import log_prediction

logger = logging.getLogger(__name__)
router = APIRouter()


class CertificateData(BaseModel):
    issuer_name: Optional[str] = ""
    recipient_name: Optional[str] = ""
    course_name: Optional[str] = ""


class SimilarityInput(BaseModel):
    cert_a: CertificateData
    cert_b: CertificateData
    method: Optional[str] = "tfidf"  # "tfidf" or "bert"


@router.post("/similarity")
async def check_similarity(
    data: SimilarityInput,
    api_key: str = Depends(verify_api_key),
):
    """Check similarity between two certificates."""
    start_time = time.time()

    cert_a = data.cert_a.model_dump()
    cert_b = data.cert_b.model_dump()

    if data.method == "bert":
        try:
            from app.models.similarity.bert_similarity import compute_semantic_similarity
            result = compute_semantic_similarity(cert_a, cert_b)
        except Exception as e:
            logger.error(f"BERT similarity failed: {e}")
            from app.models.similarity.tfidf_model import compute_similarity
            result = compute_similarity(cert_a, cert_b)
    else:
        from app.models.similarity.tfidf_model import compute_similarity
        result = compute_similarity(cert_a, cert_b)

    latency_ms = (time.time() - start_time) * 1000

    log_prediction(
        endpoint="/api/ml/similarity",
        input_data={"cert_a": cert_a, "cert_b": cert_b},
        prediction=result,
        confidence=result.get("similarity_score", 0),
        latency_ms=latency_ms,
    )

    result["latency_ms"] = round(latency_ms, 1)
    return result
