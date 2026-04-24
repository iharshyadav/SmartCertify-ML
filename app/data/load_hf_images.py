"""
load_hf_images.py — Download real certificate images from HuggingFace at build time.
Called only during Docker build via train_all.py.
Gracefully falls back to empty list if dataset unavailable.
"""
from __future__ import annotations

from typing import List
from PIL import Image


def load_authentic_images(n_max: int = 300) -> List[Image.Image]:
    """
    Load real authentic certificate images.
    Sources:
      1. agents-course/certificates  (~52 images)
      2. agents-course/final-certificates (~5 images)
    """
    images: List[Image.Image] = []

    sources = [
        ("agents-course/certificates",       "train"),
        ("agents-course/final-certificates", "train"),
    ]

    for dataset_id, split in sources:
        if len(images) >= n_max:
            break
        try:
            from datasets import load_dataset
            print(f"  Downloading {dataset_id} ({split})...")
            ds = load_dataset(dataset_id, split=split, trust_remote_code=False)
            for item in ds:
                if len(images) >= n_max:
                    break
                img = item.get("image")
                if img is None:
                    continue
                if not isinstance(img, Image.Image):
                    try:
                        img = Image.fromarray(img)
                    except Exception:
                        continue
                images.append(img.convert("RGB"))
            print(f"  Loaded {len(images)} authentic images so far")
        except Exception as e:
            print(f"  Warning: Could not load {dataset_id}: {e}")

    print(f"  Total real authentic images: {len(images)}")
    return images


def load_tampered_images(n_max: int = 150) -> List[Image.Image]:
    """
    Load real tampered images from Charles95/image_tampering.
    Falls back to empty list if unavailable.
    """
    images: List[Image.Image] = []

    try:
        from datasets import load_dataset
        print("  Downloading Charles95/image_tampering...")
        ds = load_dataset("Charles95/image_tampering", split="train",
                          trust_remote_code=False)

        # Try known column names — dataset schema may vary
        image_col = None
        for col in ["image", "tampered_image", "img", "tampered"]:
            if col in ds.column_names:
                image_col = col
                break

        if image_col is None:
            print(f"  Warning: No image column found. Columns: {ds.column_names}")
            return images

        for item in ds:
            if len(images) >= n_max:
                break
            img = item.get(image_col)
            if img is None:
                continue
            if not isinstance(img, Image.Image):
                try:
                    img = Image.fromarray(img)
                except Exception:
                    continue
            images.append(img.convert("RGB"))

        print(f"  Total real tampered images: {len(images)}")

    except Exception as e:
        print(f"  Warning: Could not load Charles95/image_tampering: {e}")
        print("  Will use synthetic tampering only.")

    return images
