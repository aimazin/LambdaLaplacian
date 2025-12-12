import numpy as np
from lambda_laplacian.transforms import laplacian_transform, graph_laplacian


def test_laplacian_1d():
    X = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    L = laplacian_transform(X)
    assert L.shape == X.shape


def test_graph_laplacian():
    adj = np.array([[0,1,0],[1,0,1],[0,1,0]])
    L = graph_laplacian(adj)
    assert L.shape == adj.shape
