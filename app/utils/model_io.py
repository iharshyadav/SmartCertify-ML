"""
SmartCertify ML — Model I/O Utilities
Save and load models using Joblib and PyTorch.
"""

import joblib
import torch
import json
import logging
from pathlib import Path
from typing import Any, Optional, Dict
from datetime import datetime, timezone

from app.config.settings import MODEL_DIR

logger = logging.getLogger(__name__)


def save_sklearn_model(model: Any, filename: str, metadata: Optional[Dict] = None) -> str:
    """Save a scikit-learn model with Joblib."""
    path = MODEL_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Saved sklearn model to {path}")

    if metadata:
        meta_path = path.with_suffix(".meta.json")
        metadata["saved_at"] = datetime.now(timezone.utc).isoformat()
        metadata["file_path"] = str(path)
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    return str(path)


def load_sklearn_model(filename: str) -> Any:
    """Load a scikit-learn model from Joblib file."""
    path = MODEL_DIR / filename
    if not path.exists():
        logger.warning(f"Model file not found: {path}")
        return None
    model = joblib.load(path)
    logger.info(f"Loaded sklearn model from {path}")
    return model


def save_pytorch_model(
    model: torch.nn.Module,
    filename: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    metadata: Optional[Dict] = None,
) -> str:
    """Save a PyTorch model checkpoint."""
    path = MODEL_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    if optimizer:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if metadata:
        checkpoint["metadata"] = metadata

    torch.save(checkpoint, path)
    logger.info(f"Saved PyTorch model to {path}")
    return str(path)


def load_pytorch_model(
    model: torch.nn.Module,
    filename: str,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    """Load a PyTorch model from checkpoint."""
    path = MODEL_DIR / filename
    if not path.exists():
        logger.warning(f"Model file not found: {path}")
        return model

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info(f"Loaded PyTorch model from {path}")
    return model


def model_exists(filename: str) -> bool:
    """Check if a model file exists."""
    return (MODEL_DIR / filename).exists()


def list_saved_models() -> Dict[str, Dict]:
    """List all saved model files with their metadata."""
    models = {}
    for path in MODEL_DIR.glob("*"):
        if path.suffix in (".joblib", ".pt", ".pth"):
            info = {
                "path": str(path),
                "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
                "modified": datetime.fromtimestamp(
                    path.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
            }
            # Check for metadata file
            meta_path = path.with_suffix(".meta.json")
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    info["metadata"] = json.load(f)
            models[path.name] = info
    return models
