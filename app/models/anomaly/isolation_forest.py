"""
SmartCertify ML — Anomaly Detection
Isolation Forest and LOF for detecting suspicious certificate patterns.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from app.config.settings import (
    RANDOM_SEED, ANOMALY_CONTAMINATION, MODEL_DIR, DATASET_PATH,
)
from app.utils.model_io import save_sklearn_model, load_sklearn_model
from app.utils.visualization import plot_anomaly_distribution
from app.config.model_registry import register_model

logger = logging.getLogger(__name__)

ANOMALY_FEATURES = [
    "issuer_reputation_score",
    "certificate_age_days",
    "metadata_completeness_score",
    "ocr_confidence_score",
    "template_match_score",
    "domain_verification_status",
    "previous_verification_count",
    "time_since_last_verification_days",
]

_model = None
_scaler = None


def train_anomaly_detector(df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Train Isolation Forest on authentic certificates only (unsupervised).
    """
    if df is None:
        df = pd.read_csv(DATASET_PATH)

    # Train on only authentic certificates
    authentic = df[df["label"] == 0].copy()
    available_features = [f for f in ANOMALY_FEATURES if f in authentic.columns]
    X = authentic[available_features].fillna(0).values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    save_sklearn_model(scaler, "anomaly_scaler.joblib")

    # Train Isolation Forest
    iso_forest = IsolationForest(
        contamination=ANOMALY_CONTAMINATION,
        n_estimators=200,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    iso_forest.fit(X_scaled)
    save_sklearn_model(iso_forest, "isolation_forest.joblib")

    # Evaluate: compute anomaly scores on full dataset
    X_full = df[available_features].fillna(0).values
    X_full_scaled = scaler.transform(X_full)
    scores = iso_forest.decision_function(X_full_scaled)
    predictions = iso_forest.predict(X_full_scaled)

    # anomaly = -1, normal = 1 in sklearn
    n_anomalies = (predictions == -1).sum()
    threshold = np.percentile(scores, ANOMALY_CONTAMINATION * 100)

    # Compare with LOF
    lof = LocalOutlierFactor(contamination=ANOMALY_CONTAMINATION, n_neighbors=20, novelty=True)
    lof.fit(X_scaled)
    lof_predictions = lof.predict(X_full_scaled)
    lof_anomalies = (lof_predictions == -1).sum()

    # Save anomaly score distribution plot
    try:
        plot_anomaly_distribution(scores, threshold)
    except Exception as e:
        logger.warning(f"Could not save anomaly plot: {e}")

    metrics = {
        "n_training_samples": len(X),
        "n_anomalies_detected": int(n_anomalies),
        "anomaly_rate": round(n_anomalies / len(X_full), 4),
        "threshold": round(float(threshold), 4),
        "lof_anomalies": int(lof_anomalies),
    }

    register_model("isolation_forest", "1.0",
                   str(MODEL_DIR / "isolation_forest.joblib"), metrics=metrics)

    logger.info(f"Anomaly detector trained: {n_anomalies} anomalies in {len(X_full)} samples")
    return metrics


def detect_anomaly(certificate_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect if a certificate is anomalous.

    Returns:
        is_anomaly, anomaly_score, anomaly_rank (LOW/MEDIUM/HIGH)
    """
    global _model, _scaler

    if _model is None:
        _model = load_sklearn_model("isolation_forest.joblib")
    if _scaler is None:
        _scaler = load_sklearn_model("anomaly_scaler.joblib")

    if _model is None:
        return {"error": "Anomaly detection model not trained"}

    # Extract features
    features = []
    for f in ANOMALY_FEATURES:
        val = certificate_data.get(f, 0)
        features.append(float(val) if val is not None else 0.0)

    X = np.array([features])
    if _scaler:
        X = _scaler.transform(X)

    # Predict
    prediction = _model.predict(X)[0]  # 1 = normal, -1 = anomaly
    anomaly_score = float(_model.decision_function(X)[0])

    is_anomaly = prediction == -1

    # Determine rank
    if anomaly_score < -0.3:
        rank = "HIGH"
    elif anomaly_score < -0.1:
        rank = "MEDIUM"
    else:
        rank = "LOW"

    return {
        "is_anomaly": bool(is_anomaly),
        "anomaly_score": round(anomaly_score, 4),
        "anomaly_rank": rank,
    }


def main():
    """Train anomaly detector."""
    print("Training anomaly detector...")
    if not Path(DATASET_PATH).exists():
        from app.data.generate_synthetic import main as gen_main
        gen_main()

    results = train_anomaly_detector()
    print(f"\n✅ Anomaly detection training complete!")
    for k, v in results.items():
        print(f"   {k}: {v}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
