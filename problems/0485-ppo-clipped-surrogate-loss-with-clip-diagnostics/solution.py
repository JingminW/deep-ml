import numpy as np

def ppo_clip_loss(old_log_probs: np.ndarray, new_log_probs: np.ndarray, advantages: np.ndarray, epsilon: float = 0.2) -> tuple:
    """
    Compute the PPO clipped surrogate loss and clip fraction diagnostic.
    
    Args:
        old_log_probs: Log-probabilities of actions under the old policy, shape (N,)
        new_log_probs: Log-probabilities of actions under the new policy, shape (N,)
        advantages: Advantage estimates for each sample, shape (N,)
        epsilon: Clipping parameter for the probability ratio
    
    Returns:
        Tuple of (loss, clip_fraction), both rounded to 4 decimal places
        - loss: Mean negated clipped surrogate objective (for minimization)
        - clip_fraction: Fraction of samples where the ratio was clipped
    """
    likelihood_ratio = np.exp(new_log_probs) / np.exp(old_log_probs)
    surrogate_objective1 = likelihood_ratio * advantages
    surrogate_objective2 = np.clip(likelihood_ratio, 1 - epsilon, 1 + epsilon) * advantages
    clipped_objective = np.minimum(surrogate_objective1, surrogate_objective2)
    
    loss = -np.mean(clipped_objective)
    clip_fraction = np.sum((surrogate_objective1 != surrogate_objective2)) / len(surrogate_objective1)

    return loss, clip_fraction 
