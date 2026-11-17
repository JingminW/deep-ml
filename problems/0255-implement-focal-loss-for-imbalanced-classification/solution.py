import numpy as np

def focal_loss(y_true, y_pred, gamma=2.0, alpha=None):
    """
    Compute Focal Loss for multi-class classification.
    
    Args:
        y_true: Ground truth labels as class indices (list or 1D array)
        y_pred: Predicted probabilities (2D array, shape: [n_samples, n_classes])
        gamma: Focusing parameter (default: 2.0)
        alpha: Class weights (optional, list or 1D array of length n_classes)
    
    Returns:
        float: Average focal loss
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

    num_classes = y_pred.shape[1]
    one_hot = np.eye(num_classes)
    y_true_one_hot = one_hot[y_true]

    cross_entropy = y_true_one_hot * np.log(y_pred)
    focal_weight = (1 - y_pred) ** gamma
    if alpha is not None:
        alpha = np.array(alpha)
        focal_loss = np.mean(np.sum(-alpha * focal_weight * cross_entropy, axis = -1))
    else:
        focal_loss = np.mean(np.sum(-focal_weight * cross_entropy, axis = -1))

    return focal_loss