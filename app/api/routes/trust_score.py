"""
trust_score.py — Issuer trust score prediction.
POST /api/ml/trust-score — Gradient Boosting Regressor.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.middleware.auth import verify_api_key
from app.models.model_store import get_trust_models

router = APIRouter()


class TrustRequest(BaseModel):
    issuer_id: str
    total_certificates_issued: Optional[int] = 100
    fraud_rate_historical: Optional[float] = 0.01
    avg_metadata_completeness: Optional[float] = 0.9
    domain_age_days: Optional[int] = 365
    verification_success_rate: Optional[float] = 0.95


def _trust_grade(score: float) -> str:
    if score >= 0.8:
        return "A"
    if score >= 0.6:
        return "B"
    if score >= 0.4:
        return "C"
    return "D"


@router.post("/trust-score")
async def get_trust_score(
    req: TrustRequest,
    _: str = Depends(verify_api_key),
):
    t0 = time.time()
    store = get_trust_models()
    feature_cols: list = store["features"]

    factors = {
        "total_certificates_issued":  float(req.total_certificates_issued or 100),
        "fraud_rate_historical":       float(req.fraud_rate_historical or 0.01),
        "avg_metadata_completeness":  float(req.avg_metadata_completeness or 0.9),
        "domain_age_days":            float(req.domain_age_days or 365),
        "verification_success_rate":  float(req.verification_success_rate or 0.95),
    }

    X = pd.DataFrame([factors])[feature_cols].fillna(0)
    raw_score = float(store["model"].predict(X)[0])
    trust_score = float(np.clip(raw_score, 0.0, 1.0))

    return {
        "issuer_id": req.issuer_id,
        "trust_score": round(trust_score, 4),
        "trust_grade": _trust_grade(trust_score),
        "factors": {k: round(v, 4) for k, v in factors.items()},
        "latency_ms": round((time.time() - t0) * 1000, 2),
    }
