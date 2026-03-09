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
from app.models.chatbot.transformer_chat import generate_response
from app.utils.monitoring import log_prediction

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatContext(BaseModel):
    certificate_data: Optional[Dict[str, Any]] = None


class ChatInput(BaseModel):
    message: str
    context: Optional[ChatContext] = None
    use_transformer: Optional[bool] = False


@router.post("/chat")
async def chat(
    data: ChatInput,
    api_key: str = Depends(verify_api_key),
):
    """AI chatbot for certificate-related queries."""
    start_time = time.time()

    context_dict = None
    if data.context:
        context_dict = data.context.model_dump()

    result = generate_response(
        message=data.message,
        context=context_dict,
        use_transformer=data.use_transformer,
    )

    latency_ms = (time.time() - start_time) * 1000

    log_prediction(
        endpoint="/api/ml/chat",
        input_data={"message": data.message},
        prediction=result,
        confidence=1.0,
        latency_ms=latency_ms,
    )

    result["latency_ms"] = round(latency_ms, 1)
    return result
