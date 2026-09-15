import numpy as np
from typing import Tuple

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Query, Key, and Value matrices.
    
    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)
    
    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    return Q, K, V

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)
    
    Returns:
        Attention output of shape (seq_len, d_k)
    """
    # Your code here
    d_k = Q.shape[1]
    
    attention = (Q @ K.T) / np.sqrt(d_k)
    attention = attention - np.max(attention, axis = 1, keepdims = True)
    attention = np.exp(attention) / np.sum(np.exp(attention), axis = 1, keepdims = True)
    output = attention @ V
    return output

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Attention output of shape (seq_len, d_model)
    """
    attention_heads = []
    Q_split = np.array_split(Q, n_heads, axis = 1)
    K_split = np.array_split(K, n_heads, axis = 1)
    V_split = np.array_split(V, n_heads, axis = 1)

    for i in range(n_heads):
        attention_heads.append(self_attention(Q_split[i], K_split[i], V_split[i]))

    output = np.concatenate(attention_heads, axis = -1)
    return output
    