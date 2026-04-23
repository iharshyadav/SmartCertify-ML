"""
ELA (Error Level Analysis) — shared image tampering utility.
Used by train_all.py and the image route.

How ELA works:
  Re-compress image at known JPEG quality.
  Tampered pixels were previously saved at a different quality level —
  they show HIGHER error than unmodified pixels.
  We extract statistical features from the error image as ML features.
"""
import io
import numpy as np
from PIL import Image, ImageChops
from typing import Tuple


def extract_ela_features(
    img: Image.Image,
    quality: int = 90,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        img:     PIL Image (any mode, converted to RGB internally)
        quality: JPEG re-compression quality level

    Returns:
        features: np.ndarray shape (12,) — [mean, std, max, p95] × 3 channels
        ela_arr:  np.ndarray shape (H, W, 3) — raw ELA image for visualisation
    """
    rgb = img.convert("RGB")

    # Re-compress at target quality
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")

    # Pixel-level difference (ELA image)
    ela_img = ImageChops.difference(rgb, recompressed)
    ela_arr = np.array(ela_img, dtype=np.float32)

    # Extract stats per channel (12 features total)
    features = []
    for ch in range(3):  # R, G, B
        ch_data = ela_arr[:, :, ch].flatten()
        features.extend([
            float(np.mean(ch_data)),
            float(np.std(ch_data)),
            float(np.max(ch_data)),
            float(np.percentile(ch_data, 95)),
        ])

    return np.array(features, dtype=np.float32), ela_arr


def get_channel_means(ela_arr: np.ndarray) -> list:
    """Returns mean ELA value per RGB channel for API response."""
    return [
        float(np.mean(ela_arr[:, :, 0])),
        float(np.mean(ela_arr[:, :, 1])),
        float(np.mean(ela_arr[:, :, 2])),
    ]
