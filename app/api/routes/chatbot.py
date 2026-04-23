"""
chatbot.py — Certificate Q&A chatbot.
POST /api/ml/chat — DistilBERT zero-shot classification.
"""
from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.middleware.auth import verify_api_key
from app.models.model_store import get_chat_model

router = APIRouter()

CANDIDATE_LABELS = [
    "verify certificate",
    "report fraud or tampering",
    "check trust score",
    "get course recommendations",
    "general help",
]

RESPONSES = {
    "verify certificate": (
        "To verify a certificate, upload it via the SmartCertify dashboard or "
        "submit the certificate ID. Our AI checks authenticity using an RF+XGB+LGB "
        "ensemble trained on 4,000 certificate records and cross-references issuer records."
    ),
    "report fraud or tampering": (
        "If you suspect a certificate is fraudulent or tampered, use the Image Analysis "
        "tool — our ResNet-18 CNN detects pixel-level modifications with high accuracy. "
        "You can also flag the certificate for manual review from the dashboard."
    ),
    "check trust score": (
        "Trust scores are computed for issuers using a Gradient Boosting model based on "
        "historical fraud rates, domain age, verification success rate, and metadata "
        "completeness. Scores range from 0 (untrusted) to 1 (fully trusted). "
        "Grade A ≥ 0.8, B ≥ 0.6, C ≥ 0.4, D < 0.4."
    ),
    "get course recommendations": (
        "SmartCertify recommends follow-up courses based on your completed certificates "
        "using BERT semantic similarity. Visit the Recommendations section in your "
        "dashboard and ensure your completed courses are listed in your profile."
    ),
    "general help": (
        "SmartCertify helps you verify, manage, and issue certificates securely. "
        "I can help you with: certificate verification, fraud & tampering detection, "
        "issuer trust scores, duplicate detection, and course recommendations. "
        "What would you like to know?"
    ),
}


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


@router.post("/chat")
async def chat(
    req: ChatRequest,
    _: str = Depends(verify_api_key),
):
    t0 = time.time()
    classifier = get_chat_model()

    result = classifier(req.message, candidate_labels=CANDIDATE_LABELS)
    top_label: str = result["labels"][0]
    top_score: float = float(result["scores"][0])

    response_text = RESPONSES.get(top_label, RESPONSES["general help"])

    return {
        "response": response_text,
        "confidence": round(top_score, 4),
        "source": f"DistilBERT zero-shot → '{top_label}'",
        "latency_ms": round((time.time() - t0) * 1000, 2),
    }
