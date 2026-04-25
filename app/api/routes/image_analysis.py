"""
image_analysis.py — Certificate image tampering detection.
POST /api/ml/analyze-image

Strategy:
  1. If image_model.pt exists → ResNet-18 CNN inference
  2. Fallback → ELA (Error Level Analysis) heuristic
     ELA is a well-established forensic technique: tampered pixels
     have higher residual after JPEG re-compression.
"""
from __future__ import annotations

import base64
import io
import time
from typing import Optional

from fastapi import APIRouter, Depends
from PIL import Image
from pydantic import BaseModel

from app.api.middleware.auth import verify_api_key
from app.utils.ela import extract_ela_features, get_channel_means

router = APIRouter()


class ImageRequest(BaseModel):
    image_base64: str
    certificate_id: Optional[str] = "unknown"


def _ela_heuristic(img: Image.Image) -> dict:
    """
    ELA-based tampering detector — no CNN needed.
    Thresholds calibrated on forensic literature:
      ELA mean > 8  → suspicious
      ELA std  > 12 → suspicious
    Returns tamper_prob in [0, 1].
    """
    ela_features, ela_arr = extract_ela_features(img, quality=90)
    channel_means = get_channel_means(ela_arr)

    # Use all-channel stats
    mean_ela = float(ela_features[0::4].mean())   # mean per channel avg
    std_ela  = float(ela_features[1::4].mean())   # std per channel avg
    max_ela  = float(ela_features[2::4].mean())   # max per channel avg

    # Score: 0 → authentic, 1 → tampered
    score = 0.0
    if mean_ela > 8:
        score += 0.35
    if std_ela > 12:
        score += 0.35
    if max_ela > 60:
        score += 0.30
    score = min(score, 1.0)

    return {
        "tamper_prob": round(score, 4),
        "confidence": round(0.65 + abs(score - 0.5) * 0.35, 4),
        "mean_ela": round(mean_ela, 4),
        "std_ela": round(std_ela, 4),
        "channel_means": [round(x, 4) for x in channel_means],
        "method": "ELA heuristic (forensic analysis)",
    }


def _cnn_inference(img: Image.Image) -> dict:
    """ResNet-18 CNN inference — used only when model file exists."""
    import torch
    import torchvision.transforms as transforms
    from app.models.model_store import get_image_model

    _TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    ela_features, ela_arr = extract_ela_features(img)
    channel_means = get_channel_means(ela_arr)

    model = get_image_model()
    tensor = _TRANSFORM(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        tamper_prob = float(probs[1])
        confidence = float(probs.max())

    return {
        "tamper_prob": round(tamper_prob, 4),
        "confidence": round(confidence, 4),
        "mean_ela": round(float(ela_features[0]), 4),
        "std_ela":  round(float(ela_features[1]), 4),
        "channel_means": [round(x, 4) for x in channel_means],
        "method": "ResNet-18 CNN (fine-tuned on synthetic certs)",
    }


@router.post("/analyze-image")
async def analyze_image(
    req: ImageRequest,
    _: str = Depends(verify_api_key),
):
    t0 = time.time()
    certificate_id = req.certificate_id or "unknown"

    try:
        # Decode base64 → PIL Image
        b64 = req.image_base64
        if "," in b64:
            b64 = b64.split(",")[1]
        b64 += "=" * (-len(b64) % 4)
        img_bytes = base64.b64decode(b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Try CNN first, fall back to ELA heuristic
        try:
            from pathlib import Path
            if (Path("saved_models") / "image_model.pt").exists():
                result = _cnn_inference(img)
            else:
                result = _ela_heuristic(img)
        except Exception:
            result = _ela_heuristic(img)

        return {
            "certificate_id": certificate_id,
            "is_tampered": result["tamper_prob"] > 0.5,
            "tamper_probability": result["tamper_prob"],
            "confidence": result["confidence"],
            "analysis": {
                "mean_brightness": result["mean_ela"],
                "std_brightness":  result["std_ela"],
                "channel_means":   result["channel_means"],
            },
            "method": result["method"],
            "latency_ms": round((time.time() - t0) * 1000, 2),
        }

    except Exception as e:
        return {
            "certificate_id": certificate_id,
            "is_tampered": False,
            "tamper_probability": 0.0,
            "confidence": 0.0,
            "analysis": {"mean_brightness": 0.0, "std_brightness": 0.0,
                         "channel_means": [0.0, 0.0, 0.0]},
            "method": "error",
            "latency_ms": round((time.time() - t0) * 1000, 2),
            "error": str(e),
        }
