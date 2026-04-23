"""
cert_image_gen.py — Generates synthetic certificate images in memory for CNN training.
No real certificate images needed — rendered entirely with PIL.
Used only by train_all.py at Docker build time.
"""
import random
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance


def make_authentic_cert(width: int = 800, height: int = 600) -> Image.Image:
    """Render a fake but visually realistic certificate image."""
    # Cream/white background
    bg_color = (
        random.randint(245, 255),
        random.randint(240, 252),
        random.randint(220, 240),
    )
    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Double border
    border_color = (
        random.randint(100, 180),
        random.randint(80, 140),
        random.randint(20, 60),
    )
    draw.rectangle([10, 10, width - 10, height - 10], outline=border_color, width=3)
    draw.rectangle([18, 18, width - 18, height - 18], outline=border_color, width=1)

    # Title block (dark coloured rectangle simulating header text)
    title_color = (
        random.randint(20, 80),
        random.randint(20, 80),
        random.randint(100, 180),
    )
    draw.rectangle([50, 40, width - 50, 100], fill=title_color)

    # Subtitle line
    sub_color = (
        random.randint(60, 120),
        random.randint(60, 120),
        random.randint(60, 120),
    )
    draw.rectangle([100, 110, width - 100, 122], fill=sub_color)

    # Body text lines (grey rectangles)
    text_color = (
        random.randint(30, 80),
        random.randint(30, 80),
        random.randint(30, 80),
    )
    line_positions = [145, 175, 210, 250, 285, 320, 360, 395, 430]
    for y in line_positions:
        w = random.randint(200, width - 120)
        x = random.randint(60, 120)
        h = random.randint(8, 14)
        draw.rectangle([x, y, x + w, y + h], fill=text_color)

    # Seal / stamp circle
    cx = random.randint(550, 680)
    cy = random.randint(420, 520)
    r = random.randint(40, 60)
    seal_color = (
        random.randint(150, 220),
        random.randint(100, 180),
        random.randint(20, 80),
    )
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=seal_color, width=3)
    draw.ellipse(
        [cx - r + 8, cy - r + 8, cx + r - 8, cy + r - 8],
        outline=seal_color, width=1,
    )

    # Signature lines
    draw.line([80, height - 80, 280, height - 80], fill=text_color, width=2)
    draw.line([width - 280, height - 80, width - 80, height - 80],
              fill=text_color, width=2)

    # Optional watermark pattern (subtle diagonal lines)
    if random.random() < 0.3:
        wm_color = (*bg_color[:2], max(0, bg_color[2] - 15))
        for i in range(0, width + height, 40):
            draw.line([(i, 0), (0, i)], fill=wm_color, width=1)

    return img


def apply_tampering(img: Image.Image) -> Image.Image:
    """Apply one or more realistic tampering operations to a certificate image."""
    img = img.copy()
    width, height = img.size
    strategy = random.choice([
        "paste_patch", "brightness_edit", "clone_stamp", "add_block",
    ])

    if strategy == "paste_patch":
        # Copy-paste a region (e.g. pasting a new name/grade)
        src_x = random.randint(0, width // 2)
        src_y = random.randint(0, height // 2)
        pw = random.randint(80, 220)
        ph = random.randint(18, 50)
        patch = img.crop((src_x, src_y, src_x + pw, src_y + ph))
        dst_x = random.randint(0, max(1, width - pw))
        dst_y = random.randint(0, max(1, height - ph))
        img.paste(patch, (dst_x, dst_y))

    elif strategy == "brightness_edit":
        # Brighten a region (simulates text erasure / lightening)
        x = random.randint(60, width - 200)
        y = random.randint(60, height - 80)
        w = random.randint(100, 250)
        h = random.randint(18, 45)
        region = img.crop((x, y, x + w, y + h))
        region = ImageEnhance.Brightness(region).enhance(random.uniform(1.8, 3.5))
        img.paste(region, (x, y))

    elif strategy == "clone_stamp":
        # Clone-stamp: duplicate small patches in multiple places
        for _ in range(random.randint(2, 5)):
            x = random.randint(0, width - 120)
            y = random.randint(0, height - 40)
            w = random.randint(60, 120)
            h = random.randint(14, 40)
            patch = img.crop((x, y, x + w, y + h))
            nx = random.randint(0, max(1, width - w))
            ny = random.randint(0, max(1, height - h))
            img.paste(patch, (nx, ny))

    elif strategy == "add_block":
        # Add solid near-white block (cover-up tamper)
        x = random.randint(60, width - 200)
        y = random.randint(100, height - 100)
        w = random.randint(80, 220)
        h = random.randint(18, 50)
        fill = (
            random.randint(238, 255),
            random.randint(238, 255),
            random.randint(228, 255),
        )
        draw = ImageDraw.Draw(img)
        draw.rectangle([x, y, x + w, y + h], fill=fill)

    return img
