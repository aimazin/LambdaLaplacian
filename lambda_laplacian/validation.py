"""Validation helpers: synthetic experiment runner."""

import numpy as np
from .imputer import LagrangianImputer
from .utils import create_missing_mask, rmse


def validate_on_synthetic(n: int = 200, missing_rate: float = 0.1, lam: float = 1.0, alpha: float = 1.0, seed: int = 0):
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 8 * np.pi, n)
    signal = np.sin(t) + 0.2 * np.cos(3 * t)
    noisy = signal + rng.normal(0, 0.3, size=signal.shape)

    mask = create_missing_mask(noisy, missing_rate=missing_rate, seed=seed)
    X_missing = noisy.copy()
    X_missing[mask] = np.nan

    imputer = LagrangianImputer(lam=lam, alpha=alpha, maxiter=500)
    imputed = imputer.fit_transform(X_missing, missing_mask=mask)

    return {
        'signal': signal,
        'noisy': noisy,
        'missing_mask': mask,
        'missing': X_missing,
        'imputed': imputed,
        'rmse_noisy': rmse(signal, noisy),
        'rmse_imputed': rmse(signal, imputed),
    }
