import numpy as np

def ema_update(ema_params, model_params_list, decay):
    """
    Compute the Exponential Moving Average of model parameters over training steps.
    
    Args:
        ema_params: numpy array, initial EMA parameters
        model_params_list: list of numpy arrays, model params at each training step
        decay: float, EMA decay rate in [0, 1]
    Returns:
        Final EMA parameters as a (nested) list, rounded to 4 decimal places
    """
    for param in model_params_list:
        ema_params = decay * ema_params + (1 - decay) * param

    return ema_params