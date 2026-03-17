from __future__ import annotations

import cv2
import numpy as np

from .types import CameraPose


def make_intrinsics(width: int, height: int, focal_factor: float = 1.2) -> np.ndarray:
    """Build a simple pinhole K matrix from image size.

    For learning projects we can approximate fx=fy from width.
    """
    f = width * focal_factor
    cx, cy = width / 2.0, height / 2.0
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]], dtype=float)


def points_from_matches(kps1, kps2, matches):
    p1 = np.float32([kps1[m.queryIdx].pt for m in matches])
    p2 = np.float32([kps2[m.trainIdx].pt for m in matches])
    return p1, p2


def estimate_relative_pose(p1, p2, K: np.ndarray):
    """Estimate relative pose using essential matrix + recoverPose."""
    E, mask = cv2.findEssentialMat(p1, p2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        return None, None, None
    inliers = mask.ravel().astype(bool)
    _, R, t, pose_mask = cv2.recoverPose(E, p1[inliers], p2[inliers], K)
    pose_inliers = pose_mask.ravel().astype(bool)
    idx = np.where(inliers)[0]
    full = np.zeros_like(inliers)
    full[idx[pose_inliers]] = True
    return R, t.reshape(3), full


def triangulate_points(
    pose_a: CameraPose,
    pose_b: CameraPose,
    p_a: np.ndarray,
    p_b: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    Pa = K @ np.hstack([pose_a.R, pose_a.t.reshape(3, 1)])
    Pb = K @ np.hstack([pose_b.R, pose_b.t.reshape(3, 1)])
    X_h = cv2.triangulatePoints(Pa, Pb, p_a.T, p_b.T)
    X = (X_h[:3] / X_h[3]).T
    return X


def solve_pnp(points3d: np.ndarray, points2d: np.ndarray, K: np.ndarray):
    if len(points3d) < 6:
        return None, None, None
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        points3d,
        points2d,
        K,
        None,
        flags=cv2.SOLVEPNP_ITERATIVE,
        reprojectionError=4.0,
        confidence=0.999,
        iterationsCount=200,
    )
    if not ok:
        return None, None, None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3), inliers
