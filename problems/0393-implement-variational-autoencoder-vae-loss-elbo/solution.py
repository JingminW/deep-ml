import numpy as np

def vae_loss(x: np.ndarray, x_reconstructed: np.ndarray, mu: np.ndarray, log_var: np.ndarray) -> tuple:
    """
    Compute the VAE loss (negative ELBO).

    Args:
        x: np.ndarray of shape (batch_size, features), original input
        x_reconstructed: np.ndarray of shape (batch_size, features), reconstructed input
        mu: np.ndarray of shape (batch_size, latent_dim), latent mean
        log_var: np.ndarray of shape (batch_size, latent_dim), latent log-variance

    Returns:
        tuple: (total_loss, reconstruction_loss, kl_divergence) as floats
    """
    kl_divergence = np.mean(0.5 * np.sum(mu ** 2 + np.exp(log_var) - 1 - log_var, axis = -1))

    recon_loss = np.mean(np.sum((x - x_reconstructed) ** 2, axis = -1))
    total_loss = recon_loss + kl_divergence

    return total_loss, recon_loss, kl_divergence
    