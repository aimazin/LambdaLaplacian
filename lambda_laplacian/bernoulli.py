"""Bernoulli-style transform placeholder.

This module provides a simple smoothing transform that resembles a Bernoulli-polynomial
style attenuation of high-frequency components. It is intentionally lightweight —
replace with a domain-specific Bernoulli transform if required.
"""

import numpy as np


def bernoulli_transform(X: np.ndarray, order: int = 2) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        if X.size < 3:
            return X
        out = X.copy()
        out[1:-1] = X[1:-1] - (X[:-2] - 2 * X[1:-1] + X[2:]) / float(max(1, order))
        return out
    else:
        out = X.copy()
        for axis in (0, 1):
            out = np.apply_along_axis(lambda a: bernoulli_transform(a, order=order), axis, out)
        return out
