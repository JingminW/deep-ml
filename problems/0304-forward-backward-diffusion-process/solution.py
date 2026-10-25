import numpy as np

def diffusion_process(x_0: np.ndarray, 
                      betas: np.ndarray,
                      timestep: int,
                      forward_noise: np.ndarray,
                      predicted_noise: np.ndarray,
                      backward_noise: np.ndarray = None) -> tuple:
    """
    Implement forward and backward diffusion processes.
    
    Args:
        x_0: Original clean data (any shape)
        betas: Noise schedule array of shape (T,)
        timestep: Current timestep t (1-indexed)
        forward_noise: Noise epsilon for forward diffusion
        predicted_noise: Model's predicted noise for backward diffusion
        backward_noise: Random noise z for stochastic backward step
    
    Returns:
        tuple: (x_t, x_t_minus_1) - noisy and denoised samples
    """
    alphas = 1 - betas
    alpha_bars = np.cumprod(alphas)
    t = timestep - 1

    alpha_t = alphas[t]
    alpha_bar_t = alpha_bars[t]
    beta_t = betas[t]

    x_t = (np.sqrt(alpha_bar_t) * x_0 + np.sqrt(1 - alpha_bar_t) * forward_noise)

    mean = (1 / np.sqrt(alpha_t)) * (x_t - (beta_t / np.sqrt(1 - alpha_bar_t)) * predicted_noise)

    if timestep == 1:
        x_t_minus_1 = mean
    else:
        alpha_bar_prev = alpha_bars[t - 1]
        posterior_var = ((1 - alpha_bar_prev) / (1 - alpha_bar_t) * beta_t)

        x_t_minus_1 = (mean + np.sqrt(posterior_var) * backward_noise)

    return x_t, x_t_minus_1