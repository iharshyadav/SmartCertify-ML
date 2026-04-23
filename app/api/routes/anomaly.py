"""
anomaly.py — Batch anomaly detection on certificate records.
POST /api/ml/anomaly — Isolation Forest.
"""
from __future__ import annotations

import time
from typing import List

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.middleware.auth import verify_api_key
from app.models.model_store import get_anomaly_models

router = APIRouter()


class AnomalyRequest(BaseModel):
    certificates: List[dict]


@router.post("/anomaly")
async def detect_anomaly(
    req: AnomalyRequest,
    _: str = Depends(verify_api_key),
):
    t0 = time.time()
    store = get_anomaly_models()
    feature_cols: list = store["features"]
    scaler = store["scaler"]
    model = store["model"]

    certs = req.certificates
    if not certs:
        return {
            "total": 0,
            "anomalies_found": 0,
            "anomaly_indices": [],
            "results": [],
            "latency_ms": round((time.time() - t0) * 1000, 2),
        }

    # Build feature matrix (fill missing with 0)
    rows = []
    for c in certs:
        row = {col: float(c.get(col, 0) or 0) for col in feature_cols}
        rows.append(row)

    X = pd.DataFrame(rows)[feature_cols].fillna(0)
    X_scaled = scaler.transform(X)

    # IsolationForest: -1 = anomaly, 1 = normal
    preds = model.predict(X_scaled)
    scores = model.score_samples(X_scaled)  # lower = more anomalous

    # Normalise anomaly score to [0, 1] where 1 = most anomalous
    scores_norm = 1.0 - (
        (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    )

    results = []
    anomaly_indices = []
    for i, (pred, score) in enumerate(zip(preds, scores_norm)):
        is_anomaly = pred == -1
        if is_anomaly:
            anomaly_indices.append(i)
        results.append({
            "index": i,
            "is_anomaly": is_anomaly,
            "anomaly_score": round(float(score), 4),
        })

    return {
        "total": len(certs),
        "anomalies_found": len(anomaly_indices),
        "anomaly_indices": anomaly_indices,
        "results": results,
        "latency_ms": round((time.time() - t0) * 1000, 2),
    }
