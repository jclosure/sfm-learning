from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

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

    # Flatten observations into arrays for efficient vectorized residuals.
    obs_img = []
    obs_pid = []
    obs_xy = []
    for pid in point_ids:
        pt = rec.points[pid]
        for ob in pt.observations:
            if ob.image_idx not in rec.poses:
                continue
            obs_img.append(ob.image_idx)
            obs_pid.append(pidx[pid])
            obs_xy.append(all_keypoints[ob.image_idx][ob.keypoint_idx].pt)

    if len(obs_img) < 20:
        return rec

    obs_img = np.asarray(obs_img, dtype=np.int32)
    obs_pid = np.asarray(obs_pid, dtype=np.int32)
    obs_xy = np.asarray(obs_xy, dtype=float)

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

    # Sparse Jacobian pattern is critical for BA speed.
    # Each observation depends only on one camera block (6 params, except fixed cam 0)
    # and one 3D point block (3 params).
    n_obs = len(obs_img)
    n_params = x0.size
    jac_sparsity = lil_matrix((2 * n_obs, n_params), dtype=np.uint8)
    for i in range(n_obs):
        img_idx = int(obs_img[i])
        if img_idx != 0:
            coff = cam_offset[img_idx]
            jac_sparsity[2 * i : 2 * i + 2, coff : coff + 6] = 1
        poff = pts_base + int(obs_pid[i]) * 3
        jac_sparsity[2 * i : 2 * i + 2, poff : poff + 3] = 1

    unique_imgs = np.unique(obs_img)

    def residuals(x):
        pts = x[pts_base:].reshape(-1, 3)

        # Decode camera parameters once per camera per call.
        cam_R = {0: rec.poses[0].R}
        cam_t = {0: rec.poses[0].t}
        for cid in free_cam_ids:
            off = cam_offset[cid]
            r = x[off : off + 3]
            t = x[off + 3 : off + 6]
            cam_R[cid] = _rodrigues_to_R(r)
            cam_t[cid] = t

        res = np.empty((n_obs, 2), dtype=float)

        for cid in unique_imgs.tolist():
            sel = obs_img == cid
            X = pts[obs_pid[sel]]
            R = cam_R[cid]
            t = cam_t[cid]

            Xc = (R @ X.T + t.reshape(3, 1)).T
            z = Xc[:, 2:3]
            z_safe = np.where(np.abs(z) < 1e-6, 1e-6, z)
            uv = (K @ Xc.T).T[:, :2] / z_safe

            r = uv - obs_xy[sel]
            bad_depth = z.ravel() <= 1e-6
            if np.any(bad_depth):
                r[bad_depth] += 50.0
            res[sel] = r

        return res.ravel()

    out = least_squares(
        residuals,
        x0,
        method="trf",
        jac_sparsity=jac_sparsity,
        loss="huber",
        f_scale=2.0,
        max_nfev=max_nfev,
        x_scale="jac",
    )
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
