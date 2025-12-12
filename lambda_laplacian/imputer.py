"""
Lagrangian-style imputer that optimizes missing entries with Laplacian smoothness.
"""

from typing import Optional
import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import laplace


class LagrangianImputer:
    """Imputer which optimizes only missing entries.

    Parameters
    ----------
    lam : float
        Smoothness (Laplacian) regularization weight.
    alpha : float
        Proximity weight to initial guess (higher -> stay closer to initial imputation).
    maxiter : int
        Maximum iterations for optimizer.
    tol : float
        Tolerance for optimizer.
    """

    def __init__(self, lam: float = 1.0, alpha: float = 1.0, maxiter: int = 1000, tol: float = 1e-6):
        self.lam = float(lam)
        self.alpha = float(alpha)
        self.maxiter = int(maxiter)
        self.tol = float(tol)

    def _flat_index(self, shape):
        return np.arange(np.prod(shape)).reshape(shape)

    def fit_transform(self, X: np.ndarray, missing_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Impute missing values in X and return imputed array.

        X may be 1D or 2D. Missing entries are NaN by default.
        """
        X = np.asarray(X, dtype=float)
        if missing_mask is None:
            missing_mask = np.isnan(X)
        else:
            missing_mask = np.asarray(missing_mask, dtype=bool)

        if X.ndim not in (1, 2):
            raise ValueError("Only 1D or 2D arrays supported")

        # initial simple imputation
        X_init = X.copy()
        if X.ndim == 1:
            mean_val = np.nanmean(X_init)
            X_init[np.isnan(X_init)] = mean_val
        else:
            col_means = np.nanmean(X_init, axis=0)
            inds = np.where(np.isnan(X_init))
            X_init[inds] = np.take(col_means, inds[1])

        missing_positions = np.argwhere(missing_mask)
        n_missing = len(missing_positions)
        if n_missing == 0:
            return X_init

        shape = X.shape
        flat_idx = self._flat_index(shape)

        x0 = X_init[tuple(missing_positions.T)]

        def objective(vars_vec):
            full = X_init.copy().flatten()
            idxs = flat_idx[tuple(missing_positions.T)]
            full[idxs] = vars_vec
            full_arr = full.reshape(shape)

            prox = np.sum((vars_vec - x0) ** 2)

            # laplacian smoothness
            try:
                L = laplace(full_arr)
            except Exception:
                if full_arr.ndim == 1:
                    L = np.zeros_like(full_arr)
                    if full_arr.size >= 3:
                        L[1:-1] = full_arr[:-2] - 2 * full_arr[1:-1] + full_arr[2:]
                else:
                    L = np.zeros_like(full_arr)

            smooth = np.sum(L ** 2)
            return self.alpha * prox + self.lam * smooth

        res = minimize(objective, x0, method="L-BFGS-B", options={"maxiter": self.maxiter, "ftol": self.tol})
        final = X_init.copy().flatten()
        idxs = flat_idx[tuple(missing_positions.T)]
        final[idxs] = res.x
        return final.reshape(shape)
