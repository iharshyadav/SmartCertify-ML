"""
SmartCertify ML — Model I/O Utilities (Lightweight)
Save and load sklearn models using joblib.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any

import joblib

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.config.settings import MODEL_DIR

logger = logging.getLogger(__name__)


def save_sklearn_model(model, filename: str, metadata: Optional[Dict] = None) -> str:
    """Save a sklearn model using joblib."""
    filepath = MODEL_DIR / filename
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, filepath)
    logger.info(f"Saved sklearn model to {filepath}")

    if metadata:
        meta_path = filepath.with_suffix(".meta.json")
        import json
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    return str(filepath)


def load_sklearn_model(filename: str):
    """Load a sklearn model from joblib."""
    filepath = MODEL_DIR / filename
    if not filepath.exists():
        logger.warning(f"Model not found: {filepath}")
        return None

    try:
        model = joblib.load(filepath)
        logger.debug(f"Loaded model from {filepath}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model {filepath}: {e}")
        return None


def model_exists(filename: str) -> bool:
    """Check if a model file exists."""
    return (MODEL_DIR / filename).exists()


def list_saved_models() -> Dict[str, Any]:
    """List all saved models with their sizes."""
    models = {}
    if MODEL_DIR.exists():
        for f in MODEL_DIR.iterdir():
            if f.suffix in (".joblib", ".pt"):
                models[f.name] = {
                    "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
                }
    return models
