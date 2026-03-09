"""
SmartCertify ML — Configuration Settings
All environment variables, paths, and constants.
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

# Create directories if they don't exist
for d in [MODEL_DIR, PLOTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── API Settings ─────────────────────────────────────────────
ML_API_KEY = os.getenv("ML_API_KEY", "smartcertify-dev-key")
API_PREFIX = "/api/ml"
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:3001").split(",")

# ─── Model Settings ──────────────────────────────────────────
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
FRAUD_RATIO = 0.08  # 8% fraudulent
DATASET_SIZE = 15000

# ─── Model File Names ────────────────────────────────────────
MODEL_FILES = {
    "preprocessor": "preprocessor.joblib",
    "fraud_rf": "fraud_rf.joblib",
    "fraud_xgb": "fraud_xgb.joblib",
    "fraud_lgbm": "fraud_lgbm.joblib",
    "fraud_ensemble": "fraud_ensemble.joblib",
    "fraud_nn": "fraud_nn.pt",
    "fraud_lr": "fraud_lr.joblib",
    "fraud_knn": "fraud_knn.joblib",
    "fraud_svm": "fraud_svm.joblib",
    "tfidf_vectorizer": "tfidf_vectorizer.joblib",
    "trust_regression": "trust_regression.joblib",
    "trust_scaler": "trust_scaler.joblib",
    "isolation_forest": "isolation_forest.joblib",
    "anomaly_scaler": "anomaly_scaler.joblib",
    "recommender": "recommender.joblib",
    "cnn_tampering": "cnn_tampering.pt",
}

# ─── Neural Network Settings ─────────────────────────────────
NN_EPOCHS = 50
NN_BATCH_SIZE = 64
NN_LEARNING_RATE = 0.001
NN_WEIGHT_DECAY = 1e-4
NN_PATIENCE = 7

# ─── Fraud Detection Thresholds ──────────────────────────────
FRAUD_THRESHOLD = 0.5
HIGH_RISK_THRESHOLD = 0.8
SIMILARITY_DUPLICATE_THRESHOLD = 0.85

# ─── Anomaly Detection ───────────────────────────────────────
ANOMALY_CONTAMINATION = 0.05

# ─── Monitoring ───────────────────────────────────────────────
PREDICTION_DB = str(LOGS_DIR / "predictions.db")
DRIFT_THRESHOLD = 0.15  # 15% shift triggers alert
DRIFT_WINDOW_SIZE = 1000

# ─── Redis ────────────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CACHE_TTL = 3600  # 1 hour

# ─── Logging ──────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ─── BERT Model ───────────────────────────────────────────────
BERT_MODEL_NAME = "all-MiniLM-L6-v2"

# ─── Dataset Path ─────────────────────────────────────────────
DATASET_PATH = str(DATA_DIR / "certificates_dataset.csv")
TRUST_DATASET_PATH = str(DATA_DIR / "trust_score_dataset.csv")
TIMESERIES_DATASET_PATH = str(DATA_DIR / "verification_timeseries.csv")
RECOMMENDATION_DATASET_PATH = str(DATA_DIR / "student_interactions.csv")
