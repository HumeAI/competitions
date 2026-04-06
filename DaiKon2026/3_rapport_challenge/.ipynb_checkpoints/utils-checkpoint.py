from __future__ import annotations
import numpy as np
from scipy.stats import pearsonr


def ccc_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if len(y_true) < 2:
        return 0.0
    mt, mp = np.mean(y_true), np.mean(y_pred)
    vt, vp = np.var(y_true), np.var(y_pred)
    cov = np.mean((y_true - mt) * (y_pred - mp))
    return float((2 * cov) / (vt + vp + (mt - mp) ** 2 + 1e-8))


def pearson_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if len(y_true) < 2:
        return 0.0
    return float(pearsonr(y_true, y_pred)[0])


def mae_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    return float(np.mean(np.abs(y_true - y_pred)))
