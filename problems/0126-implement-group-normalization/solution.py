import numpy as np

def group_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, num_groups: int, epsilon: float = 1e-5) -> np.ndarray:
    batch_size, num_channels, height, width = X.shape
    group_dim = num_channels // num_groups

    X_grouped = X.reshape(batch_size, num_groups, group_dim, height, width)
    group_means = np.mean(X_grouped, axis = (2, 3, 4), keepdims = True)
    group_vars = np.var(X_grouped, axis = (2, 3, 4), keepdims = True)
    X_norm = (X_grouped - group_means) / (np.sqrt(group_vars + epsilon))

    X_norm = X_norm.reshape((batch_size, num_channels, height, width))
    output = gamma * X_norm + beta

    return output