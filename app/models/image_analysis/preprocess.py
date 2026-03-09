"""
SmartCertify ML — Image Preprocessing
Image preprocessing utilities for the CNN tampering detector.
"""

import numpy as np
import logging
import io
import base64
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageFilter, ImageDraw, ImageFont

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

logger = logging.getLogger(__name__)

# ImageNet normalization stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TARGET_SIZE = (224, 224)


def load_image_from_base64(image_base64: str) -> Optional[Image.Image]:
    """Decode a base64 string to a PIL Image."""
    try:
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        return image.convert("RGB")
    except Exception as e:
        logger.error(f"Failed to decode base64 image: {e}")
        return None


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert a PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess an image for the CNN model.
    Resize → normalize with ImageNet stats → convert to tensor format.
    """
    # Resize
    image = image.resize(TARGET_SIZE, Image.Resampling.LANCZOS)

    # Convert to numpy
    img_array = np.array(image, dtype=np.float32) / 255.0

    # Normalize with ImageNet stats
    for i in range(3):
        img_array[:, :, i] = (img_array[:, :, i] - IMAGENET_MEAN[i]) / IMAGENET_STD[i]

    # Convert to CHW format (channels first) for PyTorch
    img_array = np.transpose(img_array, (2, 0, 1))

    return img_array


def generate_synthetic_tampered_images(n_samples: int = 100, seed: int = 42) -> list:
    """
    Generate synthetic tampered certificate images for training.
    Creates pairs: (authentic, tampered) with different corruption types.
    """
    np.random.seed(seed)
    samples = []

    for i in range(n_samples):
        # Create a synthetic "certificate" image
        img = Image.new("RGB", (400, 300), color=(255, 255, 245))
        draw = ImageDraw.Draw(img)

        # Add certificate-like content
        draw.rectangle([20, 20, 380, 280], outline=(0, 0, 0), width=2)
        draw.text((50, 40), "CERTIFICATE OF COMPLETION", fill=(0, 0, 0))
        draw.text((50, 80), f"Recipient: Student {i}", fill=(50, 50, 50))
        draw.text((50, 110), f"Course: Course {i % 20}", fill=(50, 50, 50))
        draw.text((50, 140), f"Date: 2024-{(i % 12) + 1:02d}-15", fill=(50, 50, 50))
        draw.line([(50, 220), (200, 220)], fill=(0, 0, 100), width=2)
        draw.text((50, 230), "Authorized Signature", fill=(100, 100, 100))

        # Authentic version
        authentic = img.copy()

        # Create tampered version
        tampered = img.copy()
        tampering_type = np.random.choice([
            "pixel_alter", "text_overlay", "compression", "blur", "crop_paste"
        ])

        tampered_draw = ImageDraw.Draw(tampered)

        if tampering_type == "pixel_alter":
            # Random pixel region alteration
            x1 = np.random.randint(50, 250)
            y1 = np.random.randint(50, 200)
            x2 = x1 + np.random.randint(30, 100)
            y2 = y1 + np.random.randint(20, 60)
            color = tuple(np.random.randint(200, 255, 3))
            tampered_draw.rectangle([x1, y1, x2, y2], fill=color)

        elif tampering_type == "text_overlay":
            x = np.random.randint(50, 200)
            y = np.random.randint(50, 200)
            tampered_draw.text((x, y), "MODIFIED", fill=(255, 0, 0))

        elif tampering_type == "compression":
            buffer = io.BytesIO()
            tampered.save(buffer, "JPEG", quality=5)
            buffer.seek(0)
            tampered = Image.open(buffer).convert("RGB")

        elif tampering_type == "blur":
            tampered = tampered.filter(ImageFilter.GaussianBlur(radius=3))

        elif tampering_type == "crop_paste":
            region = tampered.crop((100, 100, 200, 150))
            tampered.paste(region, (150, 150))

        samples.append({
            "authentic": preprocess_image(authentic),
            "tampered": preprocess_image(tampered),
            "tampering_type": tampering_type,
        })

    return samples
