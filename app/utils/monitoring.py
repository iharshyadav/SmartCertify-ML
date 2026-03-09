"""
SmartCertify ML — Monitoring
Prediction logging, drift detection, and performance metrics.
"""

import json
import time
import hashlib
import logging
import sqlite3
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.config.settings import PREDICTION_DB, DRIFT_THRESHOLD, DRIFT_WINDOW_SIZE, LOGS_DIR

logger = logging.getLogger(__name__)

_db_lock = threading.Lock()


def _get_db_connection() -> sqlite3.Connection:
    """Get SQLite database connection."""
    conn = sqlite3.connect(PREDICTION_DB, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            endpoint TEXT NOT NULL,
            input_hash TEXT,
            prediction TEXT,
            confidence REAL,
            latency_ms REAL,
            metadata TEXT
        )
    """)
    conn.commit()
    return conn


def log_prediction(
    endpoint: str,
    input_data: Dict[str, Any],
    prediction: Any,
    confidence: float,
    latency_ms: float,
    metadata: Optional[Dict] = None,
) -> None:
    """Log a prediction to the SQLite database."""
    try:
        input_hash = hashlib.md5(json.dumps(input_data, sort_keys=True, default=str).encode()).hexdigest()

        with _db_lock:
            conn = _get_db_connection()
            conn.execute(
                """INSERT INTO predictions (timestamp, endpoint, input_hash, prediction, confidence, latency_ms, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    endpoint,
                    input_hash,
                    json.dumps(prediction, default=str),
                    confidence,
                    latency_ms,
                    json.dumps(metadata or {}, default=str),
                ),
            )
            conn.commit()
            conn.close()
    except Exception as e:
        logger.error(f"Failed to log prediction: {e}")


def detect_drift() -> Dict[str, Any]:
    """Detect prediction drift by comparing recent predictions to baseline."""
    try:
        with _db_lock:
            conn = _get_db_connection()
            cursor = conn.execute(
                """SELECT confidence, prediction FROM predictions
                   WHERE endpoint = '/api/ml/verify'
                   ORDER BY id DESC LIMIT ?""",
                (DRIFT_WINDOW_SIZE,),
            )
            rows = cursor.fetchall()
            conn.close()

        if len(rows) < 100:
            return {"drift_detected": False, "reason": "Insufficient data", "n_samples": len(rows)}

        confidences = [r[0] for r in rows if r[0] is not None]
        predictions = []
        for r in rows:
            try:
                pred = json.loads(r[1]) if isinstance(r[1], str) else r[1]
                if isinstance(pred, dict) and "fraud_probability" in pred:
                    predictions.append(pred["fraud_probability"])
            except Exception:
                pass

        if len(predictions) < 50:
            return {"drift_detected": False, "reason": "Insufficient fraud predictions"}

        # Check drift: compare recent vs older predictions
        recent = np.array(predictions[:len(predictions)//2])
        baseline = np.array(predictions[len(predictions)//2:])

        recent_mean = np.mean(recent)
        baseline_mean = np.mean(baseline)

        drift_magnitude = abs(recent_mean - baseline_mean) / max(baseline_mean, 0.01)
        drift_detected = drift_magnitude > DRIFT_THRESHOLD

        result = {
            "drift_detected": drift_detected,
            "drift_magnitude": round(float(drift_magnitude), 4),
            "recent_fraud_mean": round(float(recent_mean), 4),
            "baseline_fraud_mean": round(float(baseline_mean), 4),
            "threshold": DRIFT_THRESHOLD,
            "n_samples": len(predictions),
        }

        # Save drift report
        if drift_detected:
            report_path = LOGS_DIR / "drift_report.json"
            result["detected_at"] = datetime.now(timezone.utc).isoformat()
            with open(report_path, "w") as f:
                json.dump(result, f, indent=2)
            logger.warning(f"Drift detected! Magnitude: {drift_magnitude:.4f}")

        return result

    except Exception as e:
        logger.error(f"Drift detection failed: {e}")
        return {"drift_detected": False, "error": str(e)}


def get_metrics() -> Dict[str, Any]:
    """Get comprehensive model performance metrics."""
    try:
        with _db_lock:
            conn = _get_db_connection()

            # Total predictions
            cursor = conn.execute("SELECT COUNT(*) FROM predictions")
            total = cursor.fetchone()[0]

            # Average confidence
            cursor = conn.execute("SELECT AVG(confidence) FROM predictions WHERE confidence IS NOT NULL")
            avg_confidence = cursor.fetchone()[0] or 0

            # Average latency
            cursor = conn.execute("SELECT AVG(latency_ms) FROM predictions WHERE latency_ms IS NOT NULL")
            avg_latency = cursor.fetchone()[0] or 0

            # Fraud detection rate
            cursor = conn.execute(
                """SELECT COUNT(*) FROM predictions
                   WHERE endpoint = '/api/ml/verify' AND confidence IS NOT NULL"""
            )
            total_verify = cursor.fetchone()[0]

            # Predictions per endpoint
            cursor = conn.execute(
                "SELECT endpoint, COUNT(*) FROM predictions GROUP BY endpoint"
            )
            per_endpoint = dict(cursor.fetchall())

            # Recent predictions (last 24h)
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
            cursor = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE timestamp > ?", (cutoff,)
            )
            recent_24h = cursor.fetchone()[0]

            conn.close()

        # Model versions from registry
        try:
            from app.config.model_registry import list_models
            model_versions = list_models()
        except Exception:
            model_versions = {}

        # Drift status
        drift = detect_drift()

        return {
            "total_predictions": total,
            "avg_confidence": round(float(avg_confidence), 4),
            "fraud_detection_count": total_verify,
            "avg_latency_ms": round(float(avg_latency), 2),
            "predictions_last_24h": recent_24h,
            "per_endpoint": per_endpoint,
            "drift_detected": drift.get("drift_detected", False),
            "model_versions": model_versions,
        }

    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        return {"error": str(e)}
