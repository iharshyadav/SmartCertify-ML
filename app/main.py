"""
SmartCertify ML Microservice — FastAPI Entry Point

A standalone Python ML microservice for the SmartCertify
blockchain-powered certificate management platform.

Run with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

import httpx

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from app.config.settings import ALLOWED_ORIGINS, API_PREFIX, LOG_LEVEL
from app.api.middleware.auth import verify_api_key
from app.api.middleware.logging import RequestLoggingMiddleware
from app.utils.model_io import list_saved_models
from app.utils.monitoring import get_metrics

# ─── Configure Logging ────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("smartcertify.main")


# ─── Keep-Alive Ping ─────────────────────────────────────────
KEEP_ALIVE_INTERVAL = int(os.getenv("KEEP_ALIVE_INTERVAL", "300"))  # seconds (default 5 min)

async def keep_alive_ping():
    """Self-ping to prevent Render free-tier from sleeping."""
    service_url = os.getenv("RENDER_EXTERNAL_URL") or os.getenv("SERVICE_URL")
    if not service_url:
        logger.info("No RENDER_EXTERNAL_URL set — keep-alive disabled (local mode)")
        return

    ping_url = f"{service_url}/health"
    logger.info(f"🏓 Keep-alive started — pinging {ping_url} every {KEEP_ALIVE_INTERVAL}s")

    async with httpx.AsyncClient(timeout=10) as client:
        while True:
            await asyncio.sleep(KEEP_ALIVE_INTERVAL)
            try:
                resp = await client.get(ping_url)
                logger.debug(f"Keep-alive ping: {resp.status_code}")
            except Exception as e:
                logger.warning(f"Keep-alive ping failed: {e}")


# ─── Lifespan (startup/shutdown) ──────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    logger.info("🚀 Starting SmartCertify ML Microservice...")

    # Pre-load critical models on startup
    try:
        from app.utils.model_io import model_exists

        loaded = []
        for name in ["preprocessor.joblib", "fraud_ensemble.joblib", "fraud_rf.joblib",
                      "isolation_forest.joblib", "trust_regression.joblib"]:
            if model_exists(name):
                loaded.append(name)

        logger.info(f"Available models: {loaded}")

    except Exception as e:
        logger.warning(f"Model preloading skipped: {e}")

    # Start keep-alive background task
    keep_alive_task = asyncio.create_task(keep_alive_ping())

    logger.info("✅ SmartCertify ML Microservice ready!")
    yield

    # Cancel keep-alive on shutdown
    keep_alive_task.cancel()
    logger.info("👋 Shutting down SmartCertify ML Microservice...")


# ─── FastAPI App ──────────────────────────────────────────────
app = FastAPI(
    title="SmartCertify ML Microservice",
    description=(
        "Machine Learning service for the SmartCertify blockchain-powered "
        "certificate management and verification platform. Provides fraud detection, "
        "similarity analysis, trust scoring, image tampering detection, anomaly detection, "
        "recommendations, and AI chatbot capabilities."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── Middleware ────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)


# ─── Import and Register Routes ───────────────────────────────
from app.api.routes import (
    fraud_detection,
    similarity,
    trust_score,
    image_analysis,
    recommendations,
    anomaly,
    chatbot,
)

app.include_router(fraud_detection.router, prefix=API_PREFIX, tags=["Fraud Detection"])
app.include_router(similarity.router, prefix=API_PREFIX, tags=["Similarity"])
app.include_router(trust_score.router, prefix=API_PREFIX, tags=["Trust Score"])
app.include_router(image_analysis.router, prefix=API_PREFIX, tags=["Image Analysis"])
app.include_router(recommendations.router, prefix=API_PREFIX, tags=["Recommendations"])
app.include_router(anomaly.router, prefix=API_PREFIX, tags=["Anomaly Detection"])
app.include_router(chatbot.router, prefix=API_PREFIX, tags=["Chatbot"])


# ─── Health & Metrics Endpoints ───────────────────────────────

@app.get("/health", tags=["System"])
async def health_check():
    """Service health check endpoint."""
    models = list_saved_models()
    return {
        "status": "ok",
        "service": "smartcertify-ml",
        "version": "1.0.0",
        "models_loaded": list(models.keys()),
        "total_models": len(models),
    }


@app.get(f"{API_PREFIX}/metrics", tags=["System"])
async def model_metrics(api_key: str = Depends(verify_api_key)):
    """Get model performance and monitoring metrics."""
    return get_metrics()


@app.get("/", tags=["System"])
async def root():
    """Root endpoint with API documentation link."""
    return {
        "service": "SmartCertify ML Microservice",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "verify": f"{API_PREFIX}/verify",
            "similarity": f"{API_PREFIX}/similarity",
            "trust_score": f"{API_PREFIX}/trust-score",
            "analyze_image": f"{API_PREFIX}/analyze-image",
            "recommend": f"{API_PREFIX}/recommend",
            "anomaly": f"{API_PREFIX}/anomaly",
            "chat": f"{API_PREFIX}/chat",
            "metrics": f"{API_PREFIX}/metrics",
        },
    }
