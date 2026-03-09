"""
SmartCertify ML — SHAP Explainability
Generate SHAP explanations for fraud detection predictions.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from app.config.settings import MODEL_DIR, PLOTS_DIR
from app.utils.model_io import load_sklearn_model

logger = logging.getLogger(__name__)

_explainer = None
_feature_names = None


def _get_explainer():
    """Load and cache the SHAP explainer."""
    global _explainer, _feature_names

    if _explainer is not None:
        return _explainer

    try:
        import shap

        model = load_sklearn_model("fraud_rf.joblib")
        if model is None:
            logger.warning("Random Forest model not found for SHAP")
            return None

        _explainer = shap.TreeExplainer(model)
        logger.info("SHAP TreeExplainer initialized with Random Forest model")
        return _explainer

    except ImportError:
        logger.warning("SHAP not installed")
        return None
    except Exception as e:
        logger.error(f"Error creating SHAP explainer: {e}")
        return None


def get_feature_names(preprocessor=None) -> List[str]:
    """Get feature names from the preprocessor."""
    global _feature_names

    if _feature_names is not None:
        return _feature_names

    if preprocessor is None:
        preprocessor = load_sklearn_model("preprocessor.joblib")

    if preprocessor is None:
        return []

    names = []
    try:
        for name, transformer, columns in preprocessor.transformers_:
            if name == "num":
                names.extend(columns)
            elif name == "text":
                # TF-IDF features
                tfidf = None
                for step_name, step in transformer.steps:
                    if hasattr(step, "get_feature_names_out"):
                        tfidf = step
                        break
                    if hasattr(step, "vocabulary_"):
                        tfidf = step
                        break

                if tfidf and hasattr(tfidf, "get_feature_names_out"):
                    names.extend([f"tfidf_{f}" for f in tfidf.get_feature_names_out()])
                elif tfidf and hasattr(tfidf, "vocabulary_"):
                    sorted_vocab = sorted(tfidf.vocabulary_.items(), key=lambda x: x[1])
                    names.extend([f"tfidf_{w}" for w, _ in sorted_vocab])
                else:
                    # Fallback
                    names.extend([f"text_feature_{i}" for i in range(500)])
    except Exception as e:
        logger.warning(f"Could not extract feature names: {e}")
        names = [f"feature_{i}" for i in range(100)]

    _feature_names = names
    return names


def explain_prediction(
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    top_n: int = 3,
) -> Dict[str, Any]:
    """
    Generate SHAP explanation for a single prediction.

    Args:
        X: Preprocessed feature array (1, n_features)
        feature_names: Optional list of feature names
        top_n: Number of top features to return

    Returns:
        Dictionary with SHAP explanation
    """
    explainer = _get_explainer()

    if explainer is None:
        # Fallback: use feature importance from Random Forest
        return _fallback_explanation(X, feature_names, top_n)

    try:
        import shap

        shap_values = explainer.shap_values(X)

        # For binary classification, shap_values may be a list [class_0, class_1]
        if isinstance(shap_values, list):
            sv = shap_values[1]  # Class 1 (fraud)
        else:
            sv = shap_values

        sv = sv.flatten()

        if feature_names is None:
            feature_names = get_feature_names()

        if len(feature_names) == 0:
            feature_names = [f"feature_{i}" for i in range(len(sv))]

        # Ensure lengths match
        min_len = min(len(sv), len(feature_names))
        sv = sv[:min_len]
        names = feature_names[:min_len]

        # Get top features by absolute SHAP value
        top_indices = np.argsort(np.abs(sv))[-top_n:][::-1]

        top_features = []
        for idx in top_indices:
            top_features.append({
                "feature": names[idx],
                "shap_value": round(float(sv[idx]), 4),
                "impact_direction": "increases_fraud" if sv[idx] > 0 else "decreases_fraud",
                "feature_value": round(float(X.flatten()[idx]), 4) if idx < len(X.flatten()) else None,
            })

        return {
            "explanation_method": "SHAP",
            "base_value": round(float(explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value), 4),
            "top_features": top_features,
        }

    except Exception as e:
        logger.error(f"SHAP explanation failed: {e}")
        return _fallback_explanation(X, feature_names, top_n)


def _fallback_explanation(
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    top_n: int = 3,
) -> Dict[str, Any]:
    """Fallback explanation using Random Forest feature importance."""
    model = load_sklearn_model("fraud_rf.joblib")
    if model is None:
        return {"explanation_method": "none", "top_features": []}

    if feature_names is None:
        feature_names = get_feature_names()

    importances = model.feature_importances_
    min_len = min(len(importances), len(feature_names))
    importances = importances[:min_len]
    names = feature_names[:min_len]

    top_indices = np.argsort(importances)[-top_n:][::-1]

    top_features = []
    for idx in top_indices:
        top_features.append({
            "feature": names[idx],
            "importance": round(float(importances[idx]), 4),
            "impact_direction": "key_factor",
            "feature_value": round(float(X.flatten()[idx]), 4) if idx < len(X.flatten()) else None,
        })

    return {
        "explanation_method": "feature_importance",
        "top_features": top_features,
    }


def generate_summary_plot(X: np.ndarray, filename: str = "shap_summary.png") -> Optional[str]:
    """Generate and save SHAP summary plot."""
    explainer = _get_explainer()
    if explainer is None:
        return None

    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values

        feature_names = get_feature_names()

        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(sv, X, feature_names=feature_names[:sv.shape[1]], show=False, max_display=15)
        path = PLOTS_DIR / filename
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP summary plot saved to {path}")
        return str(path)

    except Exception as e:
        logger.error(f"Failed to generate SHAP summary plot: {e}")
        return None
