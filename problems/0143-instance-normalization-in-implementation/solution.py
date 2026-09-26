import numpy as np

def instance_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """
    Perform Instance Normalization over a 4D tensor X of shape (B, C, H, W).
    gamma: scale parameter of shape (C,)
    beta: shift parameter of shape (C,)
    epsilon: small value for numerical stability
    Returns: normalized array of same shape as X
    """
    _, num_channels, _, _ = X.shape
    X_mean = np.mean(X, axis = (-1, -2), keepdims = True)
    X_var = np.var(X, axis = (-1, -2), keepdims = True)

    X_norm = (X - X_mean) / np.sqrt(X_var + epsilon)
    gamma = gamma.reshape(1, num_channels, 1, 1)
    beta = beta.reshape(1, num_channels, 1, 1)

    output = gamma * X_norm + beta
    return output