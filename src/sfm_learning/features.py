from __future__ import annotations

import cv2
import numpy as np


def _create_detector(name: str):
    # SIFT is usually best for SfM, ORB is fallback if SIFT unavailable.
    if name.lower() == "sift":
        if hasattr(cv2, "SIFT_create"):
            return cv2.SIFT_create(nfeatures=6000), "L2"
        name = "orb"
    if name.lower() == "orb":
        return cv2.ORB_create(nfeatures=6000), "HAMMING"
    raise ValueError(f"Unknown feature detector: {name}")


def extract_features(images: list[np.ndarray], detector: str = "sift"):
    """Extract keypoints and descriptors for every image."""
    det, norm = _create_detector(detector)
    all_kps, all_desc = [], []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps, desc = det.detectAndCompute(gray, None)
        all_kps.append(kps)
        all_desc.append(desc)
    return all_kps, all_desc, norm


def match_descriptors(desc1, desc2, norm: str, ratio: float = 0.75):
    """Lowe-ratio matching with BF matcher."""
    if desc1 is None or desc2 is None:
        return []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING if norm == "HAMMING" else cv2.NORM_L2)
    raw = bf.knnMatch(desc1, desc2, k=2)
    good = []
    for pair in raw:
        if len(pair) != 2:
            continue
        a, b = pair
        if a.distance < ratio * b.distance:
            good.append(a)
    return good
