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
from app.models.similarity.tfidf_model import find_similar
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
    method: Optional[str] = "tfidf"


@router.post("/similarity")
async def check_similarity(
    data: SimilarityInput,
    api_key: str = Depends(verify_api_key),
):
    """Check similarity between two certificates."""
    start_time = time.time()

    cert_a = data.cert_a.model_dump()
    cert_b = data.cert_b.model_dump()

    result = find_similar(
        certificate=cert_a,
        corpus=[cert_b],
        top_n=1,
        threshold=0.0,
    )

    latency_ms = (time.time() - start_time) * 1000

    # Simplify response
    similarity_score = 0.0
    if result.get("similar_certificates"):
        similarity_score = result["similar_certificates"][0].get("similarity_score", 0)

    response = {
        "similarity_score": similarity_score,
        "is_duplicate": similarity_score > 0.9,
        "method": "tfidf",
        "details": result,
    }

    log_prediction(
        endpoint="/api/ml/similarity",
        input_data={"cert_a": cert_a, "cert_b": cert_b},
        prediction=response,
        confidence=similarity_score,
        latency_ms=latency_ms,
    )

    response["latency_ms"] = round(latency_ms, 1)
    return response
