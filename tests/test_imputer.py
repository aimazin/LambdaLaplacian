import numpy as np
from lambda_laplacian.imputer import LagrangianImputer


def test_imputer_basic():
    X = np.array([1.0, np.nan, 3.0, 4.0, np.nan, 6.0])
    mask = np.isnan(X)
    imputer = LagrangianImputer(lam=0.5, alpha=1.0, maxiter=200)
    out = imputer.fit_transform(X, missing_mask=mask)
    assert out.shape == X.shape
    assert not np.any(np.isnan(out))
    assert np.all(np.isfinite(out))
