"""Shared utilities for the Influence sub-challenge."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr


EMOTIONS = [
    "anger", "anxiety", "uncertainty", "confusion", "doubt",
    "boredom", "surprise", "curiosity", "joy", "amusement",
]


def ccc_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if len(y_true) < 2:
        return 0.0
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    return float((2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-8))


def pearson_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if len(y_true) < 2:
        return 0.0
    return float(pearsonr(y_true, y_pred)[0])


def compute_influence_metrics(y_pred, y_true):
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)

    per_emotion = {}
    cccs, pears = [], []
    for i, emo in enumerate(EMOTIONS):
        ccc = ccc_score(y_true[:, i], y_pred[:, i])
        pr = pearson_score(y_true[:, i], y_pred[:, i])
        per_emotion[emo] = {"ccc": ccc, "pearson": pr}
        cccs.append(ccc)
        pears.append(pr)

    return {
        "mean_ccc": float(np.mean(cccs)),
        "mean_pearson": float(np.mean(pears)),
        "per_emotion": per_emotion,
    }
