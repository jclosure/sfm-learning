import numpy as np

from sfm_learning.geometry import make_intrinsics


def test_make_intrinsics_shape_and_center():
    K = make_intrinsics(1000, 500)
    assert K.shape == (3, 3)
    assert K[0, 2] == 500
    assert K[1, 2] == 250
    assert K[2, 2] == 1.0
