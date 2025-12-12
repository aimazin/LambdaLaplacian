"""Quickstart example for LambdaLaplacian."""

import matplotlib.pyplot as plt
from lambda_laplacian import validate_on_synthetic

if __name__ == '__main__':
    res = validate_on_synthetic(n=300, missing_rate=0.15, lam=1.0, alpha=1.0, seed=42)
    plt.plot(res['signal'], label='True')
    plt.plot(res['missing'], marker='o', linestyle='None', alpha=0.4, label='Observed (with missing)')
    plt.plot(res['imputed'], label='Imputed')
    plt.legend()
    plt.show()
    print('RMSE noisy:', res['rmse_noisy'])
    print('RMSE imputed:', res['rmse_imputed'])
