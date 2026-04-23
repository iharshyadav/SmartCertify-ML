"""
SmartCertify ML Microservice — FastAPI Entry Point
Hugging Face Spaces (Docker SDK), port 7860.
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config.settings import API_PREFIX, LOG_LEVEL

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("smartcertify.main")

# ── Redis (optional) ──────────────────────────────────────────
try:
    import redis as redis_lib
    _r = redis_lib.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
    _r.ping()
    redis_client = _r
    REDIS_AVAILABLE = True
    logger.info("Redis connected.")
except Exception:
    redis_client = None
    REDIS_AVAILABLE = False
    logger.info("Redis not available — running without cache.")

# ── Request counter (simple in-memory) ───────────────────────
_START_TIME = time.time()
_REQUEST_COUNT = 0


# ── Lifespan ──────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting SmartCertify ML Microservice (port 7860)...")
    try:
        from app.models.model_store import load_all_models
        load_all_models()
    except Exception as e:
        logger.warning(f"Model preload failed (will load lazily): {e}")
    logger.info("✅ SmartCertify ML ready!")
    yield
    logger.info("👋 Shutting down SmartCertify ML...")


# ── App ───────────────────────────────────────────────────────
app = FastAPI(
    title="SmartCertify ML",
    description=(
        "AI-powered certificate fraud detection — "
        "ResNet-18 CNN + BERT + DistilBERT + tabular ensembles."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS (must be BEFORE routers) ────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request counter middleware ────────────────────────────────
@app.middleware("http")
async def count_requests(request: Request, call_next):
    global _REQUEST_COUNT
    _REQUEST_COUNT += 1
    response = await call_next(request)
    return response


# ── Import and register routers ───────────────────────────────
from app.api.routes import (
    fraud_detection,
    similarity,
    trust_score,
    image_analysis,
    recommendations,
    anomaly,
    chatbot,
    metrics,
)

app.include_router(fraud_detection.router,  prefix=API_PREFIX, tags=["Fraud Detection"])
app.include_router(similarity.router,       prefix=API_PREFIX, tags=["Similarity"])
app.include_router(trust_score.router,      prefix=API_PREFIX, tags=["Trust Score"])
app.include_router(image_analysis.router,   prefix=API_PREFIX, tags=["Image Analysis"])
app.include_router(recommendations.router,  prefix=API_PREFIX, tags=["Recommendations"])
app.include_router(anomaly.router,          prefix=API_PREFIX, tags=["Anomaly Detection"])
app.include_router(chatbot.router,          prefix=API_PREFIX, tags=["Chatbot"])
app.include_router(metrics.router,          prefix=API_PREFIX, tags=["Metrics"])


# ── Health (no model loading — just ping) ─────────────────────
@app.get("/health", tags=["System"])
def health():
    return {
        "status": "ok",
        "service": "SmartCertify-ML",
        "version": "2.0.0",
        "port": 7860,
    }


# ── Root ─────────────────────────────────────────────────────
@app.get("/", tags=["System"])
def root():
    return {
        "service": "SmartCertify ML Microservice",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "verify":         f"{API_PREFIX}/verify",
            "analyze_image":  f"{API_PREFIX}/analyze-image",
            "similarity":     f"{API_PREFIX}/similarity",
            "trust_score":    f"{API_PREFIX}/trust-score",
            "anomaly":        f"{API_PREFIX}/anomaly",
            "chat":           f"{API_PREFIX}/chat",
            "recommend":      f"{API_PREFIX}/recommend",
            "metrics":        f"{API_PREFIX}/metrics",
        },
    }


# ── Expose counters to metrics route ─────────────────────────
def get_uptime() -> float:
    return time.time() - _START_TIME


def get_request_count() -> int:
    return _REQUEST_COUNT
