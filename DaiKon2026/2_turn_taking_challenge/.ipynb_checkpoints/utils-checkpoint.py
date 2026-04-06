"""Shared utilities for the Turn-Taking sub-challenge."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_turn_taking_metrics(speaker_preds, speaker_targets, time_preds, time_targets):
    speaker_preds = np.asarray(speaker_preds)
    speaker_targets = np.asarray(speaker_targets)
    time_preds = np.asarray(time_preds, dtype=np.float64)
    time_targets = np.asarray(time_targets, dtype=np.float64)

    macro_f1 = float(f1_score(speaker_targets, speaker_preds, average="macro"))
    acc = float(accuracy_score(speaker_targets, speaker_preds))
    mae = float(np.mean(np.abs(time_preds - time_targets)))

    return {
        "macro_f1": macro_f1,
        "accuracy": acc,
        "mae": mae,
    }
