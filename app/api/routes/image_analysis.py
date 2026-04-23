"""
image_analysis.py — Certificate image tampering detection.
POST /api/ml/analyze-image — ResNet-18 CNN (fine-tuned).
ELA stats included in analysis field for additional context.
"""
from __future__ import annotations

import base64
import io
import time
from typing import Optional

import torch
import torchvision.transforms as transforms
from fastapi import APIRouter, Depends
from PIL import Image
from pydantic import BaseModel

from app.api.middleware.auth import verify_api_key
from app.models.model_store import get_image_model
from app.utils.ela import extract_ela_features, get_channel_means

router = APIRouter()

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class ImageRequest(BaseModel):
    image_base64: str
    certificate_id: Optional[str] = "unknown"


@router.post("/analyze-image")
async def analyze_image(
    req: ImageRequest,
    _: str = Depends(verify_api_key),
):
    t0 = time.time()
    certificate_id = req.certificate_id or "unknown"

    try:
        # 1. Decode base64 → PIL Image
        b64 = req.image_base64
        if "," in b64:
            b64 = b64.split(",")[1]
        b64 += "=" * (-len(b64) % 4)  # fix padding
        img_bytes = base64.b64decode(b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # 2. ResNet-18 inference
        model = get_image_model()
        tensor = _TRANSFORM(img).unsqueeze(0)  # (1, 3, 224, 224)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0]  # [p_authentic, p_tampered]
            tamper_prob = float(probs[1])
            confidence = float(probs.max())

        # 3. ELA stats for the analysis field (supplementary visual info)
        ela_features, ela_arr = extract_ela_features(img)
        channel_means = get_channel_means(ela_arr)

        return {
            "certificate_id": certificate_id,
            "is_tampered": tamper_prob > 0.5,
            "tamper_probability": round(tamper_prob, 4),
            "confidence": round(confidence, 4),
            "analysis": {
                "mean_brightness": round(float(ela_features[0]), 4),
                "std_brightness":  round(float(ela_features[1]), 4),
                "channel_means":   [round(x, 4) for x in channel_means],
            },
            "method": "ResNet-18 CNN (fine-tuned on synthetic certs)",
            "latency_ms": round((time.time() - t0) * 1000, 2),
        }

    except Exception as e:
        return {
            "certificate_id": certificate_id,
            "is_tampered": False,
            "tamper_probability": 0.0,
            "confidence": 0.0,
            "analysis": {
                "mean_brightness": 0.0,
                "std_brightness": 0.0,
                "channel_means": [0.0, 0.0, 0.0],
            },
            "method": "error",
            "latency_ms": round((time.time() - t0) * 1000, 2),
            "error": str(e),
        }
