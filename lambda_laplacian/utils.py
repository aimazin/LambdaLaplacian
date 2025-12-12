"""Utility helpers."""

import numpy as np


def create_missing_mask(X, missing_rate: float = 0.1, seed: int = 0):
    rng = np.random.RandomState(seed)
    mask = rng.rand(*np.asarray(X).shape) < float(missing_rate)
    return mask


def rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~np.isnan(a) & ~np.isnan(b)
    if mask.sum() == 0:
        return float('nan')
    return np.sqrt(((a[mask] - b[mask]) ** 2).mean())
