from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class CameraPose:
    """Simple camera pose container.

    R maps world -> camera coordinates.
    t maps world -> camera coordinates.
    """

    R: np.ndarray
    t: np.ndarray


@dataclass(slots=True)
class Observation:
    image_idx: int
    keypoint_idx: int


@dataclass(slots=True)
class PointTrack:
    xyz: np.ndarray
    observations: list[Observation] = field(default_factory=list)


@dataclass(slots=True)
class Reconstruction:
    poses: dict[int, CameraPose]
    points: dict[int, PointTrack]
    keypoint_to_point: dict[tuple[int, int], int]
