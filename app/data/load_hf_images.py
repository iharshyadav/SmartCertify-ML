"""
load_hf_images.py — Download real certificate images from HuggingFace at build time.
Called only during Docker build via train_all.py.
Gracefully falls back to empty list if dataset unavailable.
"""
from __future__ import annotations

from typing import List
from PIL import Image


def _load_all_splits(dataset_id: str, n_max: int) -> List[Image.Image]:
    """Load images from ALL available splits of a dataset."""
    images: List[Image.Image] = []
    try:
        from datasets import load_dataset, get_dataset_split_names
        try:
            splits = get_dataset_split_names(dataset_id)
        except Exception:
            splits = ["train", "validation", "test"]

        print(f"    Available splits: {splits}")

        for split in splits:
            if len(images) >= n_max:
                break
            try:
                ds = load_dataset(dataset_id, split=split, trust_remote_code=False)
                # Find image column
                img_col = None
                for col in ["image", "img", "photo", "certificate"]:
                    if col in ds.column_names:
                        img_col = col
                        break
                if img_col is None:
                    print(f"    No image column in {split}. Cols: {ds.column_names}")
                    continue

                for item in ds:
                    if len(images) >= n_max:
                        break
                    img = item.get(img_col)
                    if img is None:
                        continue
                    if not isinstance(img, Image.Image):
                        try:
                            img = Image.fromarray(img)
                        except Exception:
                            continue
                    images.append(img.convert("RGB"))
                print(f"    Loaded {len(images)} images after split '{split}'")
            except Exception as e:
                print(f"    Skipping split '{split}': {e}")

    except Exception as e:
        print(f"    Error loading {dataset_id}: {e}")

    return images


def load_authentic_images(n_max: int = 300) -> List[Image.Image]:
    """
    Load real authentic certificate images from all available splits.
    Sources:
      1. agents-course/certificates  (train=1, validation=44, test=7)
      2. agents-course/final-certificates (validation=5)
    """
    images: List[Image.Image] = []

    sources = [
        "agents-course/certificates",
        "agents-course/final-certificates",
    ]

    for dataset_id in sources:
        if len(images) >= n_max:
            break
        print(f"  Downloading {dataset_id} (all splits)...")
        new_imgs = _load_all_splits(dataset_id, n_max - len(images))
        images.extend(new_imgs)

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

        # Try known column names
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
