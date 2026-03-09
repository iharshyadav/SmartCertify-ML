"""
SmartCertify ML — API Key Authentication Middleware
"""

import logging
from fastapi import Request, HTTPException, Security
from fastapi.security import APIKeyHeader

from app.config.settings import ML_API_KEY

logger = logging.getLogger(__name__)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    """Verify the API key from the X-API-Key header."""
    if api_key is None:
        raise HTTPException(status_code=401, detail="Missing API key. Include 'X-API-Key' header.")
    if api_key != ML_API_KEY:
        logger.warning(f"Invalid API key attempt")
        raise HTTPException(status_code=403, detail="Invalid API key.")
    return api_key
