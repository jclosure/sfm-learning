from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import least_squares

from .types import Reconstruction


def _rodrigues_to_R(v: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(v.reshape(3, 1))
    return R


def run_bundle_adjustment(
    rec: Reconstruction,
    K: np.ndarray,
    all_keypoints,
    max_nfev: int = 40,
) -> Reconstruction:
    """Tiny global BA for learning.

    Optimizes all camera extrinsics and all 3D points over available observations.
    Camera 0 is held fixed to anchor world coordinates.
    """
    if not rec.points or len(rec.poses) < 2:
        return rec

    cam_ids = sorted(rec.poses)
    if 0 not in rec.poses:
        return rec

    free_cam_ids = [c for c in cam_ids if c != 0]
    point_ids = sorted(rec.points)
    pidx = {pid: i for i, pid in enumerate(point_ids)}

    obs = []
    for pid in point_ids:
        pt = rec.points[pid]
        for ob in pt.observations:
            if ob.image_idx in rec.poses:
                obs.append((ob.image_idx, pid, ob.keypoint_idx))

    if len(obs) < 20:
        return rec

    # Initial parameter vector: [cams(6 each), points(3 each)]
    x0 = []
    for cid in free_cam_ids:
        pose = rec.poses[cid]
        rvec, _ = cv2.Rodrigues(pose.R)
        x0.extend(rvec.ravel())
        x0.extend(pose.t.ravel())
    for pid in point_ids:
        x0.extend(rec.points[pid].xyz.ravel())
    x0 = np.array(x0, dtype=float)

    cam_offset = {cid: i * 6 for i, cid in enumerate(free_cam_ids)}
    pts_base = len(free_cam_ids) * 6

    keypoints = all_keypoints

    def project(R, t, X):
        Xc = (R @ X.T + t.reshape(3, 1)).T
        u = (K @ Xc.T).T
        return u[:, :2] / u[:, 2:3]

    def residuals(x):
        res = []
        point_block = x[pts_base:]
        for img_idx, pid, kp_idx in obs:
            X = point_block[pidx[pid] * 3 : pidx[pid] * 3 + 3]
            if img_idx == 0:
                R = rec.poses[0].R
                t = rec.poses[0].t
            else:
                off = cam_offset[img_idx]
                r = x[off : off + 3]
                t = x[off + 3 : off + 6]
                R = _rodrigues_to_R(r)
            pred = project(R, t, X.reshape(1, 3))[0]
            kp = np.array(keypoints[img_idx][kp_idx].pt)
            res.extend((pred - kp).tolist())
        return np.array(res)

    out = least_squares(residuals, x0, loss="huber", f_scale=2.0, max_nfev=max_nfev)
    x = out.x

    # Write back camera params.
    for cid in free_cam_ids:
        off = cam_offset[cid]
        rec.poses[cid].R = _rodrigues_to_R(x[off : off + 3])
        rec.poses[cid].t = x[off + 3 : off + 6]

    # Write back points.
    for pid in point_ids:
        i = pts_base + pidx[pid] * 3
        rec.points[pid].xyz = x[i : i + 3]

    return rec
