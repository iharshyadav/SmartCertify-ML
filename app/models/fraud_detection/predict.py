"""
SmartCertify ML — Fraud Detection Inference
Load trained models and run predictions on new certificate data.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from app.config.settings import MODEL_DIR, FRAUD_THRESHOLD, HIGH_RISK_THRESHOLD
from app.utils.model_io import load_sklearn_model, load_pytorch_model, model_exists
from app.data.preprocess import preprocess_single
from app.models.fraud_detection.train import CertificateFraudNet

logger = logging.getLogger(__name__)

# ─── Cached model references ─────────────────────────────────
_loaded_models = {}


def _get_model(name: str):
    """Load and cache a model."""
    if name not in _loaded_models:
        if name == "neural_network":
            # Need to determine input dim from preprocessor
            preprocessor = load_sklearn_model("preprocessor.joblib")
            if preprocessor is None:
                return None
            # Estimate input dim from transformer output
            try:
                n_features = sum(
                    t[1].named_steps.get("tfidf", t[1]).get_params().get("max_features", 0)
                    if hasattr(t[1], "named_steps") and "tfidf" in t[1].named_steps
                    else len(t[2]) if isinstance(t[2], list) else 0
                    for t in preprocessor.transformers_
                )
            except Exception:
                n_features = 520  # Fallback estimate
            model = CertificateFraudNet(n_features)
            model = load_pytorch_model(model, "fraud_nn.pt")
            _loaded_models[name] = model
        else:
            filename_map = {
                "ensemble": "fraud_ensemble.joblib",
                "random_forest": "fraud_rf.joblib",
                "xgboost": "fraud_xgb.joblib",
                "lightgbm": "fraud_lgbm.joblib",
                "logistic_regression": "fraud_lr.joblib",
                "knn": "fraud_knn.joblib",
                "svm": "fraud_svm.joblib",
            }
            filename = filename_map.get(name, f"fraud_{name}.joblib")
            _loaded_models[name] = load_sklearn_model(filename)

    return _loaded_models.get(name)


def predict_fraud(
    certificate_data: Dict[str, Any],
    model_name: str = "ensemble",
) -> Dict[str, Any]:
    """
    Predict if a certificate is fraudulent.

    Args:
        certificate_data: Dictionary of certificate features
        model_name: Which model to use (default: voting ensemble)

    Returns:
        Dictionary with prediction results
    """
    # Preprocess input
    try:
        X = preprocess_single(certificate_data)
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return {"error": f"Preprocessing failed: {str(e)}"}

    # Get model
    model = _get_model(model_name)
    if model is None:
        # Fallback to any available model
        for fallback in ["ensemble", "random_forest", "xgboost", "lightgbm"]:
            model = _get_model(fallback)
            if model is not None:
                model_name = fallback
                break

    if model is None:
        return {"error": "No trained models available"}

    # Make prediction
    if model_name == "neural_network":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            fraud_probability = model(X_tensor).cpu().numpy().squeeze()
        fraud_probability = float(fraud_probability)
    else:
        if hasattr(model, "predict_proba"):
            fraud_probability = float(model.predict_proba(X)[0][1])
        else:
            fraud_probability = float(model.predict(X)[0])

    # Determine risk level
    is_authentic = fraud_probability < FRAUD_THRESHOLD
    if fraud_probability >= HIGH_RISK_THRESHOLD:
        risk_level = "CRITICAL"
    elif fraud_probability >= FRAUD_THRESHOLD:
        risk_level = "HIGH"
    elif fraud_probability >= 0.3:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # Generate risk flags
    risk_flags = _generate_risk_flags(certificate_data, fraud_probability)

    result = {
        "is_authentic": is_authentic,
        "fraud_probability": round(fraud_probability, 4),
        "confidence_score": round(1 - abs(fraud_probability - 0.5) * 2, 4) if fraud_probability != 0.5 else 0.5,
        "risk_level": risk_level,
        "risk_flags": risk_flags,
        "model_used": model_name,
    }

    return result


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

    # Check for future issue date
    if "issue_date" in data:
        try:
            import pandas as pd
            issue_dt = pd.to_datetime(data["issue_date"])
            if issue_dt > pd.Timestamp.now():
                flags.append("Future issue date detected")
        except Exception:
            pass

    if fraud_prob > HIGH_RISK_THRESHOLD:
        flags.append("Extremely high fraud probability")

    return flags


def get_loaded_models() -> List[str]:
    """Return list of currently loaded model names."""
    return list(_loaded_models.keys())


def clear_cache():
    """Clear the model cache."""
    _loaded_models.clear()
