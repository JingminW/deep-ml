import numpy as np

def forward_diffusion(x_0: np.ndarray, t: int, beta_start: float, beta_end: float, num_timesteps: int, noise: np.ndarray) -> np.ndarray:
    """
    Apply forward diffusion process to add noise to input data.
    
    Args:
        x_0: Original input data (numpy array)
        t: Timestep (1-indexed, from 1 to num_timesteps)
        beta_start: Starting value of linear beta schedule
        beta_end: Ending value of linear beta schedule
        num_timesteps: Total number of diffusion timesteps
        noise: Noise array (same shape as x_0)
    
    Returns:
        Noisy sample x_t as numpy array
    """
    timesteps = np.arange(1, num_timesteps + 1)
    betas = beta_start + (timesteps - 1) / (num_timesteps - 1) * (beta_end - beta_start)
    alphas = 1 - betas

    alpha_bars = np.cumprod(alphas)
    x_t = np.sqrt(alpha_bars[t - 1]) * x_0 + np.sqrt(1 - alpha_bars[t - 1]) * noise

    return x_t



    