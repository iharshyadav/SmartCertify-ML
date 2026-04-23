"""
fraud_detection.py — Certificate fraud detection endpoint.
POST /api/ml/verify — RF + XGB + LGB ensemble.
"""
from __future__ import annotations

import time
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.middleware.auth import verify_api_key
from app.models.model_store import get_fraud_models

router = APIRouter()


class VerifyRequest(BaseModel):
    issuer_name: str
    recipient_name: str
    course_name: str
    issue_date: str
    expiry_date: Optional[str] = None
    issuer_reputation_score: Optional[float] = 0.5
    template_match_score: Optional[float] = 0.5
    metadata_completeness_score: Optional[float] = 0.5
    domain_verification_status: Optional[int] = 1
    previous_verification_count: Optional[int] = 0


def _risk_level(prob: float) -> str:
    if prob < 0.2:
        return "LOW"
    if prob < 0.5:
        return "MEDIUM"
    if prob < 0.8:
        return "HIGH"
    return "CRITICAL"


def _build_risk_flags(req: VerifyRequest) -> List[str]:
    flags = []
    if (req.issuer_reputation_score or 0.5) < 0.3:
        flags.append("Low issuer reputation score")
    if (req.template_match_score or 0.5) < 0.4:
        flags.append("Template mismatch detected")
    if (req.metadata_completeness_score or 0.5) < 0.5:
        flags.append("Incomplete certificate metadata")
    if (req.domain_verification_status or 1) == 0:
        flags.append("Issuer domain not verified")
    if (req.previous_verification_count or 0) == 0:
        flags.append("No prior verification history")
    return flags


@router.post("/verify")
async def verify_certificate(
    req: VerifyRequest,
    _: str = Depends(verify_api_key),
):
    t0 = time.time()
    store = get_fraud_models()
    feature_cols: list = store["features"]

    # Build input feature vector (fill missing with defaults)
    input_dict = {
        "issuer_reputation_score":    req.issuer_reputation_score or 0.5,
        "template_match_score":       req.template_match_score or 0.5,
        "metadata_completeness_score": req.metadata_completeness_score or 0.5,
        "domain_verification_status": req.domain_verification_status if req.domain_verification_status is not None else 1,
        "previous_verification_count": req.previous_verification_count or 0,
        "cert_age_days":              0,
        "issuer_cert_count":          1000,
        "has_expiry":                 int(bool(req.expiry_date)),
        "name_length":                len(req.recipient_name),
        "course_name_length":         len(req.course_name),
        "total_certificates_issued":  5000,
        "fraud_rate_historical":      0.02,
        "avg_metadata_completeness":  req.metadata_completeness_score or 0.5,
        "domain_age_days":            365,
        "verification_success_rate":  0.9,
    }
    X = pd.DataFrame([input_dict])[feature_cols].fillna(0)

    # Ensemble predictions
    rf, xgb_m, lgb_m = store["rf"], store["xgb"], store["lgb"]

    rf_proba  = rf.predict_proba(X)[0]
    xgb_proba = xgb_m.predict_proba(X)[0]
    lgb_proba = lgb_m.predict_proba(X)[0]

    # Average probabilities across models
    avg_proba = (rf_proba + xgb_proba + lgb_proba) / 3.0

    # label_map: {"authentic": idx, "fake": idx, "tampered": idx}
    label_map: dict = store["label_map"]
    auth_idx = label_map.get("authentic", 0)

    # fraud_probability = P(not authentic)
    fraud_prob = float(1.0 - avg_proba[auth_idx])
    confidence = float(avg_proba.max())

    # Majority vote label
    votes = [
        rf.predict(X)[0],
        xgb_m.predict(X)[0],
        lgb_m.predict(X)[0],
    ]
    final_label_idx = max(set(votes), key=votes.count)
    inv_map = {v: k for k, v in label_map.items()}
    final_label = inv_map.get(final_label_idx, "authentic")

    return {
        "is_authentic": final_label == "authentic",
        "fraud_probability": round(fraud_prob, 4),
        "confidence_score": round(confidence, 4),
        "risk_level": _risk_level(fraud_prob),
        "risk_flags": _build_risk_flags(req),
        "model_used": "RF+XGB+LGB Ensemble",
        "latency_ms": round((time.time() - t0) * 1000, 2),
    }
