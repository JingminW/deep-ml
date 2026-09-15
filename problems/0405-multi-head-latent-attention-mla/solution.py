import numpy as np

def self_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray
):
    d_k = Q.shape[1]

    attention = (Q @ K.T) / np.sqrt(d_k)
    attention = attention - np.max(attention, axis = 1, keepdims = True)
    attention = np.exp(attention) / np.sum(np.exp(attention), axis = 1, keepdims = True)
    output = attention @ V
    return output

def multi_head_self_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    n_heads: int
):
    attention_split = []
    Q_split = np.array_split(Q, n_heads, axis = 1)
    K_split = np.array_split(K, n_heads, axis = 1)
    V_split = np.array_split(V, n_heads, axis = 1)

    for i in range(n_heads):
        attention_split.append(self_attention(Q_split[i], K_split[i], V_split[i]))

    attention = np.concatenate(attention_split, axis = 1)
    return attention



def multi_head_latent_attention(
    X: np.ndarray,
    W_dkv: np.ndarray,
    W_uk: np.ndarray,
    W_uv: np.ndarray,
    W_dq: np.ndarray,
    W_uq: np.ndarray,
    W_o: np.ndarray,
    n_heads: int
) -> tuple:
    """
    Perform Multi-Head Latent Attention (MLA).
    
    Args:
        X: Input tensor of shape (seq_len, d_model)
        W_dkv: Down-projection for KV compression (d_model, d_c_kv)
        W_uk: Up-projection for keys (d_c_kv, d_model)
        W_uv: Up-projection for values (d_c_kv, d_model)
        W_dq: Down-projection for query compression (d_model, d_c_q)
        W_uq: Up-projection for queries (d_c_q, d_model)
        W_o: Output projection (d_model, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Tuple of (output, c_kv) where:
        - output: shape (seq_len, d_model)
        - c_kv: compressed KV latent of shape (seq_len, d_c_kv)
    """
    KV_comp = X @ W_dkv
    K = KV_comp @ W_uk
    V = KV_comp @ W_uv
    Q_comp = X @ W_dq
    Q = Q_comp @ W_uq
    attention = multi_head_self_attention(Q, K, V, n_heads)
    output = attention @ W_o
    return output, KV_comp
    