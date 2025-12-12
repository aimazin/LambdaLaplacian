from lambda_laplacian.validation import validate_on_synthetic


def test_validate_synthetic():
    res = validate_on_synthetic(n=100, missing_rate=0.2, lam=1.0, alpha=1.0, seed=1)
    assert 'rmse_imputed' in res
    assert res['rmse_imputed'] <= 5.0
