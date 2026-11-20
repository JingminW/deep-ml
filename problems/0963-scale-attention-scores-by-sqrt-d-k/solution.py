import numpy as np

def scaled_attention_weights(Q: np.ndarray, K: np.ndarray) -> list:
    """
    Compute scaled dot-product attention weights.

    Args:
        Q: (n_q, d_k) query matrix
        K: (n_k, d_k) key matrix

    Returns:
        Attention weights of shape (n_q, n_k) as a nested list,
        each entry rounded to 4 decimal places.
    """
    d_k = Q.shape[1]

    scores = np.einsum("qd,kd->qk", Q, K) / np.sqrt(d_k)
    scores = scores - np.min(scores, axis = -1, keepdims = True)
    exp_scores = np.exp(scores)
    attention = exp_scores / np.sum(exp_scores, axis = -1, keepdims = True)

    return attention
