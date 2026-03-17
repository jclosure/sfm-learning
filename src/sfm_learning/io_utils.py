from __future__ import annotations

from pathlib import Path

import cv2


def load_images(image_dir: Path) -> tuple[list[str], list]:
    """Load images from a directory in sorted order."""
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in exts])
    names = [p.name for p in paths]
    images = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in paths]
    images = [im for im in images if im is not None]
    return names, images
