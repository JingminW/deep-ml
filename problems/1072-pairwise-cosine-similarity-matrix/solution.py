import numpy as np

def pairwise_cosine_similarity(X):
    X = np.array(X)
    n = X.shape[0]
    X_repeated = np.repeat(X, n, axis = 0)
    X_tiled = np.tile(X, (n, 1))

    norm_a = np.linalg.norm(X_repeated, axis = 1)
    norm_b = np.linalg.norm(X_tiled, axis = 1)
    pairwise_sum = np.sum(X_repeated * X_tiled, axis = -1) / (norm_a * norm_b)
    pairwise_sum = np.where(np.isnan(pairwise_sum), 0.0, pairwise_sum)
    return pairwise_sum.reshape(n, n)