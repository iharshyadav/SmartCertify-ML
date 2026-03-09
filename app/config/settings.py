"""
SmartCertify ML — Configuration Settings
Optimized for 512MB memory environments.
"""

import os
from pathlib import Path

# ─── Base Paths ───────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
APP_DIR = BASE_DIR / "app"
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(BASE_DIR / "saved_models")))
DATA_DIR = APP_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"
LOGS_DIR = BASE_DIR / "logs"

for d in [MODEL_DIR, PLOTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── API Settings ─────────────────────────────────────────────
ML_API_KEY = os.getenv("ML_API_KEY", "smartcertify-dev-key")
API_PREFIX = "/api/ml"
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:3001").split(",")

# ─── Model Settings (Lightweight) ────────────────────────────
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 3
FRAUD_RATIO = 0.08
DATASET_SIZE = 5000  # Reduced from 15K for memory

# ─── Fraud Detection Thresholds ──────────────────────────────
FRAUD_THRESHOLD = 0.5
HIGH_RISK_THRESHOLD = 0.8
SIMILARITY_DUPLICATE_THRESHOLD = 0.85

# ─── Anomaly Detection ───────────────────────────────────────
ANOMALY_CONTAMINATION = 0.05

# ─── Logging ──────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ─── Dataset Paths ────────────────────────────────────────────
DATASET_PATH = str(DATA_DIR / "certificates_dataset.csv")
TRUST_DATASET_PATH = str(DATA_DIR / "trust_score_dataset.csv")
TIMESERIES_DATASET_PATH = str(DATA_DIR / "verification_timeseries.csv")
RECOMMENDATION_DATASET_PATH = str(DATA_DIR / "student_interactions.csv")

# ─── Legacy settings (kept for backward compatibility) ────────
NN_BATCH_SIZE = 64
NN_LEARNING_RATE = 0.001
NN_WEIGHT_DECAY = 1e-4
