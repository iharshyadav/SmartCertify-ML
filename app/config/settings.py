"""
app/config/settings.py — SmartCertify ML configuration.
"""
import os
from pathlib import Path

# ── Base Paths ───────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
APP_DIR = BASE_DIR / "app"
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(BASE_DIR / "saved_models")))
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"
LOGS_DIR = BASE_DIR / "logs"

for d in [MODEL_DIR, PLOTS_DIR, LOGS_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── API Settings ─────────────────────────────────────────────
ML_API_KEY = os.getenv("ML_API_KEY", "smartcertify-dev-key")
API_PREFIX = "/api/ml"

# CORS: allow all origins (HF Spaces + Vercel frontend)
ALLOWED_ORIGINS = ["*"]

# ── Logging ──────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ── Fraud Detection Thresholds ────────────────────────────────
FRAUD_THRESHOLD = 0.5
HIGH_RISK_THRESHOLD = 0.8
SIMILARITY_DUPLICATE_THRESHOLD = 0.85

# ── Anomaly Detection ─────────────────────────────────────────
ANOMALY_CONTAMINATION = 0.1

# ── Random Seed ───────────────────────────────────────────────
RANDOM_SEED = 42
