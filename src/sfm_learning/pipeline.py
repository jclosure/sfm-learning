from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ba import run_bundle_adjustment
from .features import extract_features, match_descriptors
from .geometry import (
    estimate_relative_pose,
    make_intrinsics,
    points_from_matches,
    solve_pnp,
    triangulate_points,
)
from .io_utils import load_images
from .types import CameraPose, Observation, PointTrack, Reconstruction


@dataclass(slots=True)
class PipelineResult:
    reconstruction: Reconstruction
    image_names: list[str]
    intrinsics: np.ndarray


def run_pipeline(
    image_dir: Path,
    detector: str = "sift",
    ratio: float = 0.75,
    run_ba: bool = True,
) -> PipelineResult:
    """Run a small incremental SfM pipeline over an image folder.

    Learning goals:
    - keep code readable over clever
    - keep each geometric step explicit
    """
    image_names, images = load_images(image_dir)
    if len(images) < 2:
        raise ValueError("Need at least 2 images")

    H, W = images[0].shape[:2]
    K = make_intrinsics(W, H)

    keypoints, descriptors, norm = extract_features(images, detector=detector)

    # Precompute pairwise matches for all image pairs.
    pair_matches: dict[tuple[int, int], list] = {}
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            pair_matches[(i, j)] = match_descriptors(descriptors[i], descriptors[j], norm, ratio)

    # Choose initialization pair with max match count.
    init_pair = max(pair_matches, key=lambda k: len(pair_matches[k]))
    i0, i1 = init_pair
    m01 = pair_matches[init_pair]
    p0, p1 = points_from_matches(keypoints[i0], keypoints[i1], m01)
    R01, t01, inlier_mask = estimate_relative_pose(p0, p1, K)
    if R01 is None:
        raise RuntimeError("Could not initialize reconstruction")

    rec = Reconstruction(poses={}, points={}, keypoint_to_point={})
    rec.poses[i0] = CameraPose(np.eye(3), np.zeros(3))
    rec.poses[i1] = CameraPose(R01, t01)

    # Build initial 3D points from inlier correspondences.
    pid = 0
    inlier_matches = [m for k, m in enumerate(m01) if inlier_mask[k]]
    p0_in, p1_in = points_from_matches(keypoints[i0], keypoints[i1], inlier_matches)
    X = triangulate_points(rec.poses[i0], rec.poses[i1], p0_in, p1_in, K)

    for k, m in enumerate(inlier_matches):
        rec.points[pid] = PointTrack(
            xyz=X[k],
            observations=[Observation(i0, m.queryIdx), Observation(i1, m.trainIdx)],
        )
        rec.keypoint_to_point[(i0, m.queryIdx)] = pid
        rec.keypoint_to_point[(i1, m.trainIdx)] = pid
        pid += 1

    registered = {i0, i1}
    unregistered = set(range(len(images))) - registered

    # Incremental registration loop.
    while unregistered:
        best_img = None
        best_corr = None

        for img_idx in list(unregistered):
            pts3d, pts2d = [], []
            kp_ref = []
            # Collect 2D-3D correspondences from matches with registered images.
            for ridx in registered:
                a, b = sorted((ridx, img_idx))
                matches = pair_matches.get((a, b), [])
                for m in matches:
                    r_kp = m.queryIdx if a == ridx else m.trainIdx
                    i_kp = m.trainIdx if a == ridx else m.queryIdx
                    p_id = rec.keypoint_to_point.get((ridx, r_kp))
                    if p_id is not None:
                        pts3d.append(rec.points[p_id].xyz)
                        pts2d.append(keypoints[img_idx][i_kp].pt)
                        kp_ref.append((i_kp, p_id))

            if len(pts3d) < 20:
                continue
            R, t, inliers = solve_pnp(np.array(pts3d, float), np.array(pts2d, float), K)
            if R is None or inliers is None:
                continue
            score = len(inliers)
            if best_img is None or score > len(best_corr[2]):
                best_img = img_idx
                best_corr = (R, t, inliers, kp_ref)

        if best_img is None:
            break

        R, t, inliers, kp_ref = best_corr
        rec.poses[best_img] = CameraPose(R, t)
        registered.add(best_img)
        unregistered.remove(best_img)

        # Attach confirmed inlier observations to existing points.
        for idx in inliers.ravel().tolist():
            kp_idx, point_id = kp_ref[idx]
            key = (best_img, kp_idx)
            if key not in rec.keypoint_to_point:
                rec.keypoint_to_point[key] = point_id
                rec.points[point_id].observations.append(Observation(best_img, kp_idx))

        # Triangulate new points with every other registered camera.
        for ridx in sorted(registered):
            if ridx == best_img:
                continue
            a, b = sorted((ridx, best_img))
            matches = pair_matches.get((a, b), [])
            if not matches:
                continue

            p_a, p_b, m_used = [], [], []
            for m in matches:
                r_kp = m.queryIdx if a == ridx else m.trainIdx
                i_kp = m.trainIdx if a == ridx else m.queryIdx
                if (best_img, i_kp) in rec.keypoint_to_point:
                    continue
                if (ridx, r_kp) in rec.keypoint_to_point:
                    continue
                p_a.append(keypoints[ridx][r_kp].pt)
                p_b.append(keypoints[best_img][i_kp].pt)
                m_used.append((r_kp, i_kp))

            if len(m_used) < 12:
                continue

            Xa = np.array(p_a, float)
            Xb = np.array(p_b, float)
            Xnew = triangulate_points(rec.poses[ridx], rec.poses[best_img], Xa, Xb, K)

            for row, (r_kp, i_kp) in enumerate(m_used):
                xyz = Xnew[row]
                if not np.isfinite(xyz).all() or np.linalg.norm(xyz) > 1e5:
                    continue
                rec.points[pid] = PointTrack(
                    xyz=xyz,
                    observations=[Observation(ridx, r_kp), Observation(best_img, i_kp)],
                )
                rec.keypoint_to_point[(ridx, r_kp)] = pid
                rec.keypoint_to_point[(best_img, i_kp)] = pid
                pid += 1

    if run_ba:
        rec = run_bundle_adjustment(rec, K, keypoints)

    return PipelineResult(reconstruction=rec, image_names=image_names, intrinsics=K)
