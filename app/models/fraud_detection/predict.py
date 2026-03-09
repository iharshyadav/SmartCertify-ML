"""
SmartCertify ML — Fraud Detection Inference (Lightweight)
Load trained sklearn models and run predictions on new certificate data.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from app.config.settings import MODEL_DIR, FRAUD_THRESHOLD, HIGH_RISK_THRESHOLD
from app.utils.model_io import load_sklearn_model, model_exists
from app.data.preprocess import preprocess_single

logger = logging.getLogger(__name__)

_loaded_models = {}


def _get_model(name: str):
    """Load and cache a model."""
    if name not in _loaded_models:
        filename_map = {
            "ensemble": "fraud_ensemble.joblib",
            "random_forest": "fraud_rf.joblib",
            "xgboost": "fraud_xgb.joblib",
            "lightgbm": "fraud_lgbm.joblib",
            "logistic_regression": "fraud_lr.joblib",
        }
        filename = filename_map.get(name, f"fraud_{name}.joblib")
        _loaded_models[name] = load_sklearn_model(filename)

    return _loaded_models.get(name)


def predict_fraud(
    certificate_data: Dict[str, Any],
    model_name: str = "ensemble",
) -> Dict[str, Any]:
    """Predict if a certificate is fraudulent."""
    try:
        X = preprocess_single(certificate_data)
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return {"error": f"Preprocessing failed: {str(e)}"}

    model = _get_model(model_name)
    if model is None:
        for fallback in ["ensemble", "random_forest", "xgboost", "lightgbm", "logistic_regression"]:
            model = _get_model(fallback)
            if model is not None:
                model_name = fallback
                break

    if model is None:
        return {"error": "No trained models available"}

    if hasattr(model, "predict_proba"):
        fraud_probability = float(model.predict_proba(X)[0][1])
    else:
        fraud_probability = float(model.predict(X)[0])

    is_authentic = fraud_probability < FRAUD_THRESHOLD
    if fraud_probability >= HIGH_RISK_THRESHOLD:
        risk_level = "CRITICAL"
    elif fraud_probability >= FRAUD_THRESHOLD:
        risk_level = "HIGH"
    elif fraud_probability >= 0.3:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    risk_flags = _generate_risk_flags(certificate_data, fraud_probability)

    return {
        "is_authentic": is_authentic,
        "fraud_probability": round(fraud_probability, 4),
        "confidence_score": round(1 - abs(fraud_probability - 0.5) * 2, 4),
        "risk_level": risk_level,
        "risk_flags": risk_flags,
        "model_used": model_name,
    }


def _generate_risk_flags(data: Dict[str, Any], fraud_prob: float) -> List[str]:
    """Generate human-readable risk flags based on features."""
    flags = []

    rep = data.get("issuer_reputation_score", 1.0)
    if isinstance(rep, (int, float)) and rep < 0.3:
        flags.append("Low issuer reputation score")

    tmpl = data.get("template_match_score", 1.0)
    if isinstance(tmpl, (int, float)) and tmpl < 0.4:
        flags.append("Low template match score")

    meta = data.get("metadata_completeness_score", 1.0)
    if isinstance(meta, (int, float)) and meta < 0.3:
        flags.append("Incomplete metadata")

    domain = data.get("domain_verification_status", 1)
    if domain == 0:
        flags.append("Domain verification failed")

    verif = data.get("previous_verification_count", 1)
    if isinstance(verif, (int, float)) and verif == 0:
        flags.append("Never previously verified")

    if fraud_prob > HIGH_RISK_THRESHOLD:
        flags.append("Extremely high fraud probability")

    return flags


def get_loaded_models() -> List[str]:
    return list(_loaded_models.keys())


def clear_cache():
    _loaded_models.clear()
