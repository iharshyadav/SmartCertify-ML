"""
SmartCertify ML — Request/Response Logging Middleware
"""

import time
import logging
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("smartcertify.api")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all API requests and responses."""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # Log request
        logger.info(
            f"→ {request.method} {request.url.path} "
            f"[{request.client.host if request.client else 'unknown'}]"
        )

        # Process request
        try:
            response = await call_next(request)
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(
                f"✗ {request.method} {request.url.path} — "
                f"ERROR: {str(e)} [{duration_ms:.1f}ms]"
            )
            raise

        duration_ms = (time.time() - start_time) * 1000

        # Log response
        log_fn = logger.warning if response.status_code >= 400 else logger.info
        log_fn(
            f"← {request.method} {request.url.path} — "
            f"{response.status_code} [{duration_ms:.1f}ms]"
        )

        # Add timing header
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.1f}"
        return response
