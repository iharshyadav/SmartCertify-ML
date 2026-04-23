"""
metrics.py — Model health and performance metrics.
GET /api/ml/metrics
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from app.api.middleware.auth import verify_api_key

router = APIRouter()

MODELS_LOADED = [
    "fraud_rf",
    "fraud_xgb",
    "fraud_lgb",
    "resnet18_cnn",
    "trust_model",
    "anomaly_model",
    "sentence_transformers",
    "distilbert_zero_shot",
]


@router.get("/metrics")
async def get_metrics(_: str = Depends(verify_api_key)):
    from app.main import get_uptime, get_request_count

    return {
        "status": "ok",
        "service": "SmartCertify-ML",
        "version": "2.0.0",
        "models_loaded": MODELS_LOADED,
        "total_models": len(MODELS_LOADED),
        "uptime_seconds": round(get_uptime(), 2),
        "total_requests": get_request_count(),
    }
