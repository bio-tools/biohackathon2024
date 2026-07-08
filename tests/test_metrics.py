import numpy as np

from bh24_literature_mining.config import ID2LABEL
from bh24_literature_mining.evaluation.metrics import compute_metrics


def test_compute_metrics_perfect():
    predictions = np.array([[[0, 0, 10], [10, 0, 0], [0, 0, 10]]])  # pred: O, B-BT, O
    labels = np.array([[2, 0, 2]])  # true: O, B-BT, O
    result = compute_metrics((predictions, labels), ID2LABEL)
    assert set(result) == {"precision", "recall", "f1", "accuracy"}
    assert result["f1"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0


def test_compute_metrics_ignores_minus_100():
    predictions = np.array([[[0, 0, 10], [10, 0, 0], [0, 0, 10]]])
    labels = np.array([[2, -100, 2]])  # middle token masked
    result = compute_metrics((predictions, labels), ID2LABEL)
    assert set(result) == {"precision", "recall", "f1", "accuracy"}
