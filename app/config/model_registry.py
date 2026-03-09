"""
SmartCertify ML — Model Registry
Track model versions, metadata, and performance metrics.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from app.config.settings import MODEL_DIR, LOGS_DIR


REGISTRY_PATH = LOGS_DIR / "model_registry.json"


def _load_registry() -> Dict[str, Any]:
    """Load the model registry from disk."""
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, "r") as f:
            return json.load(f)
    return {"models": {}, "last_updated": None}


def _save_registry(registry: Dict[str, Any]) -> None:
    """Save the model registry to disk."""
    registry["last_updated"] = datetime.now(timezone.utc).isoformat()
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def register_model(
    model_name: str,
    version: str,
    file_path: str,
    metrics: Optional[Dict[str, float]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Register a trained model with its metadata and metrics."""
    registry = _load_registry()

    entry = {
        "version": version,
        "file_path": file_path,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics or {},
        "metadata": metadata or {},
        "is_active": True,
    }

    if model_name not in registry["models"]:
        registry["models"][model_name] = {"versions": []}

    # Deactivate previous versions
    for v in registry["models"][model_name]["versions"]:
        v["is_active"] = False

    registry["models"][model_name]["versions"].append(entry)
    registry["models"][model_name]["active_version"] = version

    _save_registry(registry)
    return entry


def get_active_model(model_name: str) -> Optional[Dict[str, Any]]:
    """Get the currently active version of a model."""
    registry = _load_registry()
    if model_name not in registry["models"]:
        return None

    for v in reversed(registry["models"][model_name]["versions"]):
        if v.get("is_active", False):
            return v
    return None


def list_models() -> Dict[str, Any]:
    """List all registered models and their active versions."""
    registry = _load_registry()
    summary = {}
    for name, data in registry["models"].items():
        active = get_active_model(name)
        summary[name] = {
            "active_version": data.get("active_version"),
            "total_versions": len(data["versions"]),
            "metrics": active["metrics"] if active else {},
        }
    return summary


def get_all_metrics() -> Dict[str, Dict[str, float]]:
    """Get performance metrics for all active models."""
    registry = _load_registry()
    metrics = {}
    for name, data in registry["models"].items():
        active = get_active_model(name)
        if active and active.get("metrics"):
            metrics[name] = active["metrics"]
    return metrics
