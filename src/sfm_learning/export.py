from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from .types import Reconstruction


def export_point_cloud(rec: Reconstruction, out_path: Path) -> Path:
    """Export sparse points as PLY."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pts = np.array([p.xyz for p in rec.points.values()], dtype=float)
    if len(pts) == 0:
        raise ValueError("No points to export")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.io.write_point_cloud(str(out_path), pcd)
    return out_path
