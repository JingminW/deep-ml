import numpy as np

def he_initialization(n_in: int, n_out: int, mode: str = 'fan_in', distribution: str = 'normal', seed: int = None) -> np.ndarray:
    """
    Implement He (Kaiming) weight initialization.
    
    Parameters:
    n_in: number of input units
    n_out: number of output units
    mode: 'fan_in' or 'fan_out'
    distribution: 'normal' or 'uniform'
    seed: random seed for reproducibility
    
    Returns:
    numpy array of shape (n_in, n_out) with He-initialized weights
    """
    # rng = np.random.default_rng(seed)

    # fan = n_in if mode == "fan_in" else n_out
    # if distribution == "normal":
    #     std = np.sqrt(2.0 / fan)
    #     weights = rng.normal(loc = 0.0, scale = std, size = (n_in, n_out))
    # else:
    #     bound = np.sqrt(6.0 / fan)
    #     weights = rng.uniform(low = -bound, high = bound, size = (n_in, n_out))

    # return weights
    if seed is not None:
        np.random.seed(seed)

    fan = n_in if mode == "fan_in" else n_out

    if distribution == "normal":
        std = np.sqrt(2.0 / fan)
        weights = np.random.normal(
            loc=0.0,
            scale=std,
            size=(n_in, n_out)
        )
    else:
        bound = np.sqrt(6.0 / fan)
        weights = np.random.uniform(
            low=-bound,
            high=bound,
            size=(n_in, n_out)
        )

    return weights
    