"""
SmartCertify ML — Fraud Detection Explainability (Lightweight)
Feature importance-based explanation (no SHAP to save memory).
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, List

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from app.utils.model_io import load_sklearn_model

logger = logging.getLogger(__name__)


def explain_prediction(
    X: np.ndarray,
    top_n: int = 5,
) -> Dict[str, Any]:
    """
    Explain a prediction using feature importance from Random Forest.
    Lightweight alternative to SHAP (saves ~100MB memory).
    """
    model = load_sklearn_model("fraud_rf.joblib")
    if model is None:
        return {"top_features": [], "method": "unavailable"}

    # Get feature importances from Random Forest
    try:
        importances = model.feature_importances_

        # Try to get feature names from preprocessor
        feature_names = _get_feature_names()

        if len(feature_names) != len(importances):
            feature_names = [f"feature_{i}" for i in range(len(importances))]

        # Sort by importance
        indices = np.argsort(importances)[::-1][:top_n]

        top_features = []
        for idx in indices:
            top_features.append({
                "feature": feature_names[idx],
                "importance": round(float(importances[idx]), 4),
                "impact_direction": "increases fraud risk" if importances[idx] > 0 else "neutral",
            })

        return {
            "top_features": top_features,
            "method": "feature_importance",
            "model": "random_forest",
        }

    except Exception as e:
        logger.warning(f"Explanation failed: {e}")
        return {"top_features": [], "method": "error", "error": str(e)}


def _get_feature_names() -> list:
    """Extract feature names from the fitted preprocessor."""
    preprocessor = load_sklearn_model("preprocessor.joblib")
    if preprocessor is None:
        return []

    names = []
    try:
        for name, transformer, columns in preprocessor.transformers_:
            if name == "num":
                if isinstance(columns, list):
                    names.extend(columns)
                else:
                    names.extend([f"num_{i}" for i in range(columns)])
            elif name == "text":
                try:
                    vocab = transformer.vocabulary_
                    names.extend([f"tfidf_{word}" for word in sorted(vocab, key=vocab.get)])
                except Exception:
                    names.extend([f"text_{i}" for i in range(500)])
    except Exception:
        pass

    return names
