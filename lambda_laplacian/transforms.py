"""
Laplacian and graph utilities.
"""

import numpy as np
from scipy.sparse import csgraph


def laplacian_transform(X: np.ndarray) -> np.ndarray:
    """Discrete Laplacian: 1D second-difference or 2D five-point stencil.

    Returns an array of same shape with Laplacian values.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        out = np.zeros_like(X)
        if X.size >= 3:
            out[1:-1] = X[:-2] - 2 * X[1:-1] + X[2:]
        return out
    elif X.ndim == 2:
        out = np.zeros_like(X)
        out[1:-1, 1:-1] = (
            X[:-2, 1:-1] + X[2:, 1:-1] + X[1:-1, :-2] + X[1:-1, 2:] - 4 * X[1:-1, 1:-1]
        )
        return out
    else:
        raise ValueError("Only 1D or 2D arrays supported")


def graph_laplacian(adj: np.ndarray) -> np.ndarray:
    """Compute unnormalized graph Laplacian from adjacency matrix.
    """
    adj = np.asarray(adj, dtype=float)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("Adjacency must be a square matrix")
    return csgraph.laplacian(adj, normed=False)
