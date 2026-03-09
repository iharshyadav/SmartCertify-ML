"""
SmartCertify ML — Recommendations API Route
POST /api/ml/recommend
"""

import time
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.middleware.auth import verify_api_key
from app.models.recommendation.recommender import get_recommendations
from app.utils.monitoring import log_prediction

logger = logging.getLogger(__name__)
router = APIRouter()


class RecommendationInput(BaseModel):
    student_id: str
    completed_courses: List[str] = []
    n_recommendations: Optional[int] = 5


@router.post("/recommend")
async def recommend_certificates(
    data: RecommendationInput,
    api_key: str = Depends(verify_api_key),
):
    """Get certificate/course recommendations for a student."""
    start_time = time.time()

    result = get_recommendations(
        student_id=data.student_id,
        completed_courses=data.completed_courses,
        n_recommendations=data.n_recommendations,
    )

    latency_ms = (time.time() - start_time) * 1000

    log_prediction(
        endpoint="/api/ml/recommend",
        input_data=data.model_dump(),
        prediction=result,
        confidence=1.0,
        latency_ms=latency_ms,
    )

    result["latency_ms"] = round(latency_ms, 1)
    return result
