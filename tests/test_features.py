import numpy as np

from sfm_learning.features import match_descriptors


def test_match_descriptors_handles_none():
    out = match_descriptors(None, np.ones((5, 32), dtype=np.uint8), norm="HAMMING")
    assert out == []
