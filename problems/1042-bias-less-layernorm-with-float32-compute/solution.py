import numpy as np

def biasless_layernorm(x, gamma, eps=1e-5):
    """
    Bias-less LayerNorm over the last dimension with float32 compute.

    Args:
        x: numpy array of shape (..., D)
        gamma: numpy array of shape (D,)
        eps: float, numerical stability constant

    Returns:
        numpy array of same shape and dtype as x
    """
    x_mean = np.mean(x, axis = -1, keepdims = True)
    x_var = np.var(x, axis = -1, keepdims = True)
    x = (x - x_mean) / np.sqrt(x_var + eps)
    
    return gamma * x
