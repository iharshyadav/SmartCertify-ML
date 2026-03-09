"""
SmartCertify ML — Monitoring (Lightweight)
Prediction logging using file-based JSON (no SQLite for reduced memory).
"""

import json
import time
import hashlib
import logging
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.config.settings import LOGS_DIR

logger = logging.getLogger(__name__)

_log_lock = threading.Lock()
LOG_FILE = LOGS_DIR / "predictions.jsonl"


def log_prediction(
    endpoint: str,
    input_data: Dict[str, Any],
    prediction: Any,
    confidence: float,
    latency_ms: float,
    metadata: Optional[Dict] = None,
) -> None:
    """Log a prediction to a JSONL file."""
    try:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "endpoint": endpoint,
            "confidence": confidence,
            "latency_ms": round(latency_ms, 1),
        }

        with _log_lock:
            with open(LOG_FILE, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")

    except Exception as e:
        logger.error(f"Failed to log prediction: {e}")


def detect_drift() -> Dict[str, Any]:
    """Simplified drift detection."""
    return {"drift_detected": False, "reason": "Lightweight mode — drift check disabled"}


def get_metrics() -> Dict[str, Any]:
    """Get basic metrics from the prediction log."""
    try:
        total = 0
        if LOG_FILE.exists():
            with open(LOG_FILE) as f:
                total = sum(1 for _ in f)

        try:
            from app.config.model_registry import list_models
            model_versions = list_models()
        except Exception:
            model_versions = {}

        return {
            "total_predictions": total,
            "model_versions": model_versions,
            "mode": "lightweight",
        }

    except Exception as e:
        return {"error": str(e)}
