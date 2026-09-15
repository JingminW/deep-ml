import numpy as np

def gated_attention(
    X: np.ndarray,
    W_q: np.ndarray,
    W_k: np.ndarray,
    W_v: np.ndarray,
    W_g: np.ndarray
) -> np.ndarray:
    """
    Compute Gated Attention output.
    
    Args:
        X: Input tensor of shape (seq_len, d_model)
        W_q: Query projection of shape (d_model, d_k)
        W_k: Key projection of shape (d_model, d_k)
        W_v: Value projection of shape (d_model, d_v)
        W_g: Gate projection of shape (d_model, d_v)
    
    Returns:
        Gated attention output of shape (seq_len, d_v), rounded to 4 decimal places
    
    Hint: First compute standard scaled dot-product attention, then apply
    a sigmoid gate to modulate the output.
    """
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    g = X @ W_g

    d_k = Q.shape[1]
    attention = (Q @ K.T) / np.sqrt(d_k)
    attention = attention - np.max(attention, axis = 1, keepdims = True)
    denominator = np.sum(np.exp(attention), axis = 1, keepdims = True)
    attention = np.exp(attention) / denominator

    output = attention @ V
    sigmoid_gate = 1 / (1 + np.exp(-g))
    return sigmoid_gate * output
