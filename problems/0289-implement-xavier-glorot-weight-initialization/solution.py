import numpy as np

def xavier_initialization(shape: tuple, mode: str = 'uniform', seed: int = None) -> np.ndarray:
    """
    Implement Xavier/Glorot weight initialization.
    
    Args:
        shape: Tuple of (fan_in, fan_out) representing weight matrix dimensions
        mode: 'uniform' or 'normal' initialization
        seed: Random seed for reproducibility (optional)
    
    Returns:
        Initialized weight matrix as numpy array
    """
    if seed is not None:
        np.random.seed(seed)

    fan_in, fan_out = shape

    if mode == "normal":
        std = np.sqrt(2 / (fan_in + fan_out))
        weights = np.random.normal(loc = 0, scale = std, size = (fan_in, fan_out))
    elif mode == "uniform":
        bound = np.sqrt(6 / (fan_in + fan_out))
        weights = np.random.uniform(-bound, bound, size = (fan_in, fan_out))

    return weights