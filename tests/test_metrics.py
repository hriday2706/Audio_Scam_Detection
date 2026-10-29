"""Unit tests for metric computations and confidence intervals."""

import numpy as np
import pytest

from voxshield.models.evaluate import compute_bootstrap_ci, compute_metrics


def test_compute_metrics_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.9, 0.8])
    m = compute_metrics(y_true, y_prob, threshold=0.5)

    assert m["accuracy"] == 1.0
    assert m["f1_scam"] == 1.0
    assert m["precision_scam"] == 1.0
    assert m["recall_scam"] == 1.0
    assert m["pr_auc"] == 1.0
    assert m["roc_auc"] == 1.0
    assert m["sample_count"] == 4


def test_compute_metrics_imperfect():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.2, 0.3, 0.8, 0.7])  # 1 FP, 1 FN
    m = compute_metrics(y_true, y_prob, threshold=0.5)

    assert m["accuracy"] == 0.5
    assert 0.0 < m["f1_scam"] < 1.0
    assert m["confusion_matrix"] == [[1, 1], [1, 1]]


def test_compute_bootstrap_ci():
    y_true = np.array([0] * 50 + [1] * 50)
    y_prob = np.array([0.1] * 45 + [0.8] * 5 + [0.2] * 5 + [0.9] * 45)

    ci = compute_bootstrap_ci(y_true, y_prob, threshold=0.5, n_bootstraps=100)
    assert "f1_scam_ci" in ci
    assert "pr_auc_ci" in ci
    f1_low, f1_high = ci["f1_scam_ci"]
    assert 0.0 <= f1_low <= f1_high <= 1.0
