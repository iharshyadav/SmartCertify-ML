"""
SmartCertify ML — Image Analysis (Lightweight)
Simple pixel-level analysis without CNN (no PyTorch needed).
"""

import numpy as np
import logging
import io
import base64
from pathlib import Path
from typing import Dict, Any

from PIL import Image, ImageStat

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

logger = logging.getLogger(__name__)


def load_image_from_base64(image_base64: str):
    """Decode a base64 string to a PIL Image."""
    try:
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        return image.convert("RGB")
    except Exception as e:
        logger.error(f"Failed to decode base64 image: {e}")
        return None


def analyze_image(
    image_input,
    certificate_id: str = "unknown",
) -> Dict[str, Any]:
    """
    Analyze a certificate image for tampering using pixel statistics.
    Lightweight alternative to CNN — no PyTorch needed.
    """
    if isinstance(image_input, str):
        image = load_image_from_base64(image_input)
    elif isinstance(image_input, Image.Image):
        image = image_input
    else:
        return {"error": "Invalid image input"}

    if image is None:
        return {"error": "Failed to load image"}

    # Pixel-level analysis
    stat = ImageStat.Stat(image)
    img_array = np.array(image, dtype=np.float32) / 255.0

    # Metrics for tampering detection
    mean_brightness = float(np.mean(img_array))
    std_brightness = float(np.std(img_array))
    channel_means = [float(m) for m in stat.mean]
    channel_stds = [float(s) for s in stat.stddev]

    # Simple heuristic scoring
    tampering_score = 0.0

    # Very low or very high contrast can indicate tampering
    if std_brightness < 0.05 or std_brightness > 0.35:
        tampering_score += 0.3

    # Uneven channel distribution
    channel_range = max(channel_means) - min(channel_means)
    if channel_range > 80:
        tampering_score += 0.2

    # Very uniform regions (compression artifacts)
    uniform_pixels = float(np.mean(np.abs(np.diff(img_array, axis=1)) < 0.01))
    if uniform_pixels > 0.7:
        tampering_score += 0.2

    tampering_score = min(tampering_score, 1.0)
    is_tampered = tampering_score > 0.4

    return {
        "certificate_id": certificate_id,
        "is_tampered": is_tampered,
        "tamper_probability": round(tampering_score, 4),
        "confidence": round(1.0 - tampering_score if not is_tampered else tampering_score, 4),
        "analysis": {
            "mean_brightness": round(mean_brightness, 4),
            "std_brightness": round(std_brightness, 4),
            "channel_means": [round(m, 2) for m in channel_means],
        },
        "method": "pixel_statistics",
    }
