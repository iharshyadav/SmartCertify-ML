"""
model_store.py — Singleton model registry.
All models loaded once at startup via lru_cache.
All routes import from here — never call joblib.load() per request.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models
import joblib
from pathlib import Path
from functools import lru_cache

MODEL_DIR = Path("saved_models")
DEVICE = torch.device("cpu")


def _load(filename: str):
    path = MODEL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}. "
            "Rebuild the Docker image to retrain all models."
        )
    return joblib.load(path)


# ── Fraud Detection (tabular ensemble) ───────────────────────

@lru_cache(maxsize=1)
def get_fraud_models():
    return {
        "rf":        _load("fraud_rf.pkl"),
        "xgb":       _load("fraud_xgb.pkl"),
        "lgb":       _load("fraud_lgb.pkl"),
        "features":  _load("fraud_features.pkl"),
        "label_map": _load("fraud_label_map.pkl"),
    }


# ── Image Tampering (ResNet-18 CNN) ───────────────────────────

@lru_cache(maxsize=1)
def get_image_model() -> nn.Module:
    """Load ResNet-18 fine-tuned for binary tamper classification."""
    m = tv_models.resnet18(weights=None)
    m.fc = nn.Sequential(
        nn.Linear(m.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2),
    )
    state_path = MODEL_DIR / "image_model.pt"
    if not state_path.exists():
        raise FileNotFoundError(
            f"image_model.pt not found at {state_path}. "
            "Rebuild Docker image to retrain."
        )
    state = torch.load(str(state_path), map_location=DEVICE)
    m.load_state_dict(state)
    m.eval()
    m.to(DEVICE)
    return m


# ── Semantic Similarity (sentence-transformers) ───────────────

@lru_cache(maxsize=1)
def get_similarity_model():
    """Load sentence-transformers model from local HF cache."""
    from sentence_transformers import SentenceTransformer
    name_file = MODEL_DIR / "similarity_model_name.txt"
    model_name = name_file.read_text().strip() if name_file.exists() else "all-MiniLM-L6-v2"
    return SentenceTransformer(model_name)


# ── Chat (DistilBERT zero-shot) ───────────────────────────────

@lru_cache(maxsize=1)
def get_chat_model():
    """Load DistilBERT zero-shot classifier from local HF cache."""
    from transformers import pipeline
    name_file = MODEL_DIR / "chat_model_name.txt"
    model_name = (name_file.read_text().strip() if name_file.exists()
                  else "typeform/distilbert-base-uncased-mnli")
    return pipeline(
        "zero-shot-classification",
        model=model_name,
        device=-1,  # CPU
    )


# ── Trust Score (Gradient Boosting) ──────────────────────────

@lru_cache(maxsize=1)
def get_trust_models():
    return {
        "model":    _load("trust_model.pkl"),
        "features": _load("trust_features.pkl"),
    }


# ── Anomaly Detection (Isolation Forest) ─────────────────────

@lru_cache(maxsize=1)
def get_anomaly_models():
    return {
        "model":    _load("anomaly_model.pkl"),
        "scaler":   _load("anomaly_scaler.pkl"),
        "features": _load("anomaly_features.pkl"),
    }


# ── Preload all at startup ────────────────────────────────────

def load_all_models() -> None:
    """Preload all models into lru_cache at startup."""
    print("Preloading all models into memory...")
    get_fraud_models();     print("  ✓ fraud models (RF+XGB+LGB)")
    get_image_model();      print("  ✓ ResNet-18 CNN")
    get_similarity_model(); print("  ✓ sentence-transformers")
    get_chat_model();       print("  ✓ DistilBERT zero-shot")
    get_trust_models();     print("  ✓ trust model (GBR)")
    get_anomaly_models();   print("  ✓ anomaly model (IsoForest)")
    print("All models ready.")
