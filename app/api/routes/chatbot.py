"""
SmartCertify ML — Chatbot API Route
POST /api/ml/chat
"""

import time
import logging
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.middleware.auth import verify_api_key
from app.models.chatbot.transformer_chat import chat as chatbot_respond
from app.utils.monitoring import log_prediction

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatInput(BaseModel):
    message: str
    session_id: Optional[str] = "default"


@router.post("/chat")
async def chat_endpoint(
    data: ChatInput,
    api_key: str = Depends(verify_api_key),
):
    """AI chatbot for certificate-related queries."""
    start_time = time.time()

    result = chatbot_respond(
        query=data.message,
        session_id=data.session_id,
    )

    latency_ms = (time.time() - start_time) * 1000

    log_prediction(
        endpoint="/api/ml/chat",
        input_data={"message": data.message},
        prediction=result,
        confidence=result.get("confidence", 0),
        latency_ms=latency_ms,
    )

    result["latency_ms"] = round(latency_ms, 1)
    return result
