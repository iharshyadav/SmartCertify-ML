"""
similarity.py — Certificate duplicate detection.
POST /api/ml/similarity — BERT sentence-transformers cosine similarity.
"""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

from app.api.middleware.auth import verify_api_key
from app.models.model_store import get_similarity_model

router = APIRouter()


class SimilarityRequest(BaseModel):
    cert_a: dict  # {issuer_name?, recipient_name?, course_name?}
    cert_b: dict


def _cert_to_text(c: dict) -> str:
    parts = [
        c.get("issuer_name", ""),
        c.get("recipient_name", ""),
        c.get("course_name", ""),
    ]
    return " ".join(p for p in parts if p).strip() or "unknown"


@router.post("/similarity")
async def check_similarity(
    req: SimilarityRequest,
    _: str = Depends(verify_api_key),
):
    t0 = time.time()
    model = get_similarity_model()

    text_a = _cert_to_text(req.cert_a)
    text_b = _cert_to_text(req.cert_b)

    # Encode both texts → (1, 384) embeddings
    emb_a = model.encode([text_a])
    emb_b = model.encode([text_b])

    score = float(cosine_similarity(emb_a, emb_b)[0][0])
    # Clamp to [0, 1] (cosine can be slightly negative for very different texts)
    score = max(0.0, min(1.0, score))

    return {
        "similarity_score": round(score, 4),
        "is_duplicate": score > 0.85,
        "method": "BERT sentence-transformers (all-MiniLM-L6-v2)",
        "latency_ms": round((time.time() - t0) * 1000, 2),
    }
