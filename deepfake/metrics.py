"""Métriques d'évaluation, en NumPy pur (aucune dépendance à scikit-learn).

Deux niveaux sont calculés :

* **frame** — chaque image est un exemple ; c'est ce qui était rapporté avant ;
* **vidéo** — moyenne des probabilités par vidéo, puis une décision par vidéo.

Les frames d'une même vidéo sont quasi identiques : le niveau frame surestime
systématiquement la performance réelle. Le niveau vidéo est celui à présenter.
"""

from __future__ import annotations

import math
from typing import Hashable, Iterable, Sequence

import numpy as np

POS_LABEL = 1  # FAKE


def roc_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """AUC par la statistique de Mann-Whitney, avec gestion correcte des ex aequo."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = int((y_true == POS_LABEL).sum())
    n_neg = int((y_true != POS_LABEL).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=float)

    # Moyenne des rangs pour les valeurs égales.
    _, inverse, counts = np.unique(y_score, return_inverse=True, return_counts=True)
    rank_sums = np.zeros(len(counts), dtype=float)
    np.add.at(rank_sums, inverse, ranks)
    ranks = (rank_sums / counts)[inverse]

    return float((ranks[y_true == POS_LABEL].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def confusion(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, int]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return {
        "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
    }


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def classification_metrics(
    y_true: Sequence[int], y_prob: Sequence[float], threshold: float = 0.5
) -> dict[str, float]:
    """Accuracy, précision/rappel/F1 par classe et AUC pour un seuil donné."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = (np.asarray(y_prob, dtype=float) >= threshold).astype(int)
    counts = confusion(y_true, y_pred)

    precision_fake = _safe_div(counts["tp"], counts["tp"] + counts["fp"])
    recall_fake = _safe_div(counts["tp"], counts["tp"] + counts["fn"])
    precision_real = _safe_div(counts["tn"], counts["tn"] + counts["fn"])
    recall_real = _safe_div(counts["tn"], counts["tn"] + counts["fp"])

    f1_fake = _safe_div(2 * precision_fake * recall_fake, precision_fake + recall_fake)
    f1_real = _safe_div(2 * precision_real * recall_real, precision_real + recall_real)

    return {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()),
        "precision_fake": precision_fake,
        "recall_fake": recall_fake,
        "f1_fake": f1_fake,
        "precision_real": precision_real,
        "recall_real": recall_real,
        "f1_real": f1_real,
        "auc": roc_auc(y_true, y_prob),
        **{f"count_{key}": value for key, value in counts.items()},
    }


def best_threshold(
    y_true: Sequence[int], y_prob: Sequence[float], criterion: str = "youden"
) -> float:
    """Seuil optimal calculé **sur la validation**, jamais sur le test.

    ``youden`` maximise sensibilité + spécificité - 1 ; ``f1`` maximise le F1 FAKE.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(np.unique(y_true)) < 2:
        return 0.5

    candidates = np.unique(np.clip(y_prob, 0.0, 1.0))
    candidates = np.unique(np.concatenate(([0.0], candidates, [1.0])))
    best_score, best_value = -np.inf, 0.5

    for threshold in candidates:
        metrics = classification_metrics(y_true, y_prob, float(threshold))
        if criterion == "youden":
            score = metrics["recall_fake"] + metrics["recall_real"] - 1.0
        elif criterion == "f1":
            score = metrics["f1_fake"]
        else:
            raise ValueError(f"Critère inconnu : {criterion!r}")
        if score > best_score:
            best_score, best_value = score, float(threshold)

    return best_value


def aggregate_by_video(
    groups: Iterable[Hashable],
    y_true: Sequence[int],
    y_prob: Sequence[float],
) -> dict[Hashable, tuple[int, float]]:
    """Moyenne les probabilités par vidéo et vérifie la cohérence des labels."""
    buckets: dict[Hashable, list[float]] = {}
    labels: dict[Hashable, int] = {}

    for group, label, prob in zip(groups, y_true, y_prob):
        buckets.setdefault(group, []).append(float(prob))
        label = int(label)
        if group in labels and labels[group] != label:
            raise ValueError(f"La vidéo {group!r} porte deux labels différents.")
        labels[group] = label

    return {group: (labels[group], float(np.mean(probs))) for group, probs in buckets.items()}


def evaluate_video_level(
    groups: Iterable[Hashable],
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: float = 0.5,
) -> tuple[dict[str, float], list[tuple[Hashable, int, float]]]:
    """Métriques au niveau vidéo (une décision par vidéo)."""
    aggregated = aggregate_by_video(groups, y_true, y_prob)
    items = [(group, label, prob) for group, (label, prob) in aggregated.items()]
    labels = [label for _, label, _ in items]
    probs = [prob for _, _, prob in items]
    return classification_metrics(labels, probs, threshold), items


def bootstrap_ci(
    metric,
    y_true: Sequence[int],
    y_score: Sequence[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Intervalle de confiance percentile par bootstrap (rééchantillonnage des images)."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    rng = np.random.default_rng(seed)
    values = []
    n = len(y_true)

    for _ in range(n_boot):
        indices = rng.integers(0, n, n)
        if len(np.unique(y_true[indices])) < 2:
            continue
        values.append(metric(y_true[indices], y_score[indices]))

    if not values:
        return (float("nan"), float("nan"))
    low, high = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


def mcnemar_test(
    y_true: Sequence[int], pred_a: Sequence[int], pred_b: Sequence[int]
) -> dict[str, float]:
    """Test de McNemar exact : les deux modèles diffèrent-ils significativement ?

    ``a_only`` = A a raison et B a tort ; ``b_only`` = l'inverse.
    """
    y_true = np.asarray(y_true, dtype=int)
    correct_a = np.asarray(pred_a, dtype=int) == y_true
    correct_b = np.asarray(pred_b, dtype=int) == y_true
    a_only = int((correct_a & ~correct_b).sum())
    b_only = int((~correct_a & correct_b).sum())
    discordant = a_only + b_only

    if discordant == 0:
        return {"a_only": 0, "b_only": 0, "p_value": 1.0, "statistic": 0.0}

    # p-value exacte bilatérale (loi binomiale sous H0, p = 0.5).
    tail = sum(math.comb(discordant, k) for k in range(0, min(a_only, b_only) + 1))
    p_value = min(1.0, 2.0 * tail / 2.0**discordant)
    statistic = (abs(a_only - b_only) - 1) ** 2 / discordant

    return {
        "a_only": a_only,
        "b_only": b_only,
        "p_value": float(p_value),
        "statistic": float(statistic),
    }


def expected_calibration_error(
    y_true: Sequence[int], y_prob: Sequence[float], n_bins: int = 10
) -> float:
    """ECE : écart moyen entre confiance annoncée et fréquence observée."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)

    total = len(y_true)
    if total == 0:
        return float("nan")

    error = 0.0
    for index in range(n_bins):
        mask = indices == index
        if not mask.any():
            continue
        confidence = float(y_prob[mask].mean())
        frequency = float(y_true[mask].mean())
        error += mask.sum() / total * abs(confidence - frequency)
    return float(error)


def reliability_curve(
    y_true: Sequence[int], y_prob: Sequence[float], n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Points (confiance moyenne, fréquence observée, effectif) pour un diagramme de fiabilité."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)

    confidence, frequency, counts = [], [], []
    for index in range(n_bins):
        mask = indices == index
        if not mask.any():
            continue
        confidence.append(float(y_prob[mask].mean()))
        frequency.append(float(y_true[mask].mean()))
        counts.append(int(mask.sum()))
    return np.array(confidence), np.array(frequency), np.array(counts)
