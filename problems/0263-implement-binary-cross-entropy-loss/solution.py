import torch

def binary_cross_entropy(y_true: torch.Tensor, y_pred: torch.Tensor, epsilon: float = 1e-15) -> float:
    """
    Compute binary cross-entropy loss.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted probabilities (between 0 and 1)
        epsilon: Small value for numerical stability
    
    Returns:
        Mean binary cross-entropy loss
    """
    y_pred = torch.clip(y_pred, epsilon, 1 - epsilon)
    loss = y_true * torch.log(y_pred) + (1 - y_true) * torch.log((1 - y_pred))
    mean_loss = -torch.mean(loss)
    return mean_loss.item()