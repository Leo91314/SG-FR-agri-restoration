import numpy as np
import pytest

from cea_plus.metrics import boundary_f_score


def _edge_case_image() -> np.ndarray:
    image = np.zeros((32, 32, 3), dtype=np.float32)
    image[:, 16:] = 1.0
    image[10:22, 10:22, 1] = 0.5
    return image


def test_boundary_f_score_accepts_explicit_edge_quantile():
    gt = _edge_case_image()
    pred = gt.copy()
    pred[:, 17:] = 0.95

    default_score = boundary_f_score(gt, pred)
    explicit_score = boundary_f_score(gt, pred, edge_quantile=0.82)
    lower_threshold_score = boundary_f_score(gt, pred, edge_quantile=0.70)

    assert explicit_score == default_score
    assert 0.0 <= lower_threshold_score <= 1.0


def test_boundary_f_score_rejects_invalid_edge_quantile():
    gt = _edge_case_image()

    with pytest.raises(ValueError, match="edge_quantile"):
        boundary_f_score(gt, gt, edge_quantile=1.0)
