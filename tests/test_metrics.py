"""Tests des métriques (NumPy pur, aucune dépendance lourde)."""

from __future__ import annotations

import numpy as np
import pytest

from deepfake import metrics


def test_roc_auc_perfect_and_inverted():
    y = np.array([0, 0, 1, 1])
    assert metrics.roc_auc(y, [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert metrics.roc_auc(y, [0.9, 0.8, 0.2, 0.1]) == pytest.approx(0.0)


def test_roc_auc_ties_are_half_credit():
    y = np.array([0, 1])
    assert metrics.roc_auc(y, [0.5, 0.5]) == pytest.approx(0.5)


def test_roc_auc_single_class_is_nan():
    assert np.isnan(metrics.roc_auc([0, 0, 0], [0.1, 0.2, 0.3]))


def test_classification_metrics_counts_and_scores():
    y_true = [0, 0, 1, 1, 1]
    y_prob = [0.1, 0.6, 0.7, 0.8, 0.2]
    result = metrics.classification_metrics(y_true, y_prob, threshold=0.5)
    assert result["count_tn"] == 1
    assert result["count_fp"] == 1
    assert result["count_fn"] == 1
    assert result["count_tp"] == 2
    assert result["accuracy"] == pytest.approx(3 / 5)
    assert result["precision_fake"] == pytest.approx(2 / 3)
    assert result["recall_fake"] == pytest.approx(2 / 3)
    assert result["f1_fake"] == pytest.approx(2 / 3)


def test_best_threshold_separates_perfectly():
    y_true = [0, 0, 1, 1]
    y_prob = [0.05, 0.1, 0.9, 0.95]
    threshold = metrics.best_threshold(y_true, y_prob, "youden")
    assert 0.1 <= threshold <= 0.9


def test_aggregate_by_video_averages_probabilities():
    groups = ["a", "a", "b"]
    labels = [1, 1, 0]
    probs = [0.2, 0.4, 0.9]
    aggregated = metrics.aggregate_by_video(groups, labels, probs)
    assert aggregated["a"] == (1, pytest.approx(0.3))
    assert aggregated["b"] == (0, pytest.approx(0.9))


def test_aggregate_by_video_rejects_conflicting_labels():
    with pytest.raises(ValueError, match="deux labels"):
        metrics.aggregate_by_video(["a", "a"], [0, 1], [0.1, 0.9])


def test_video_level_metrics_on_tiny_example():
    groups = ["v1", "v1", "v2", "v2"]
    labels = [1, 1, 0, 0]
    probs = [0.9, 0.8, 0.2, 0.1]
    result, items = metrics.evaluate_video_level(groups, labels, probs, 0.5)
    assert result["accuracy"] == 1.0
    assert len(items) == 2


def test_mcnemar_symmetric_case_is_not_significant():
    y_true = [1, 1, 0, 0, 1, 0]
    pred_a = [1, 0, 0, 1, 1, 0]
    pred_b = [0, 1, 1, 0, 1, 0]
    result = metrics.mcnemar_test(y_true, pred_a, pred_b)
    assert result["a_only"] == 2
    assert result["b_only"] == 2
    assert result["p_value"] == pytest.approx(1.0)


def test_mcnemar_identical_predictions():
    y_true = [1, 0, 1]
    result = metrics.mcnemar_test(y_true, y_true, y_true)
    assert result["p_value"] == 1.0
    assert result["a_only"] == 0


def test_bootstrap_ci_brackets_the_estimate():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 200)
    y_prob = np.clip(0.5 + 0.3 * (y_true - 0.5) + rng.normal(0, 0.1, 200), 0, 1)
    low, high = metrics.bootstrap_ci(metrics.roc_auc, y_true, y_prob, n_boot=100)
    assert low < high
    assert metrics.roc_auc(y_true, y_prob) == pytest.approx((low + high) / 2, abs=0.05)


def test_ece_zero_for_perfectly_confident_correct_predictions():
    y_true = [0, 1, 0, 1]
    y_prob = [0.0, 1.0, 0.0, 1.0]
    assert metrics.expected_calibration_error(y_true, y_prob) == pytest.approx(0.0)


def test_reliability_curve_lengths_match():
    y_true = [0, 1, 0, 1]
    y_prob = [0.05, 0.95, 0.4, 0.6]
    confidence, frequency, counts = metrics.reliability_curve(y_true, y_prob)
    assert len(confidence) == len(frequency) == len(counts)
