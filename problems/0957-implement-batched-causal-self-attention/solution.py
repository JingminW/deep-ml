import numpy as np

def batched_causal_self_attention(X, W_query, W_key, W_value):
    """
    Batched causal self-attention.

    Args:
        X: nested list of shape (B, T, d_in)
        W_query, W_key, W_value: nested lists of shape (d_in, d_out)

    Returns:
        Nested list of shape (B, T, d_out) -- the context vectors.
    """
    Q = np.einsum("bld,dk->blk", X, W_query)
    K = np.einsum("bld,dk->blk", X, W_key)
    V = np.einsum("bld,dk->blk", X, W_value)
    batch_size, seq_len, d_k = Q.shape

    scores = np.einsum("bqd,bkd->bqk", Q, K) / np.sqrt(d_k)
    matrix = np.ones((batch_size, seq_len, seq_len)) * -np.inf
    mask = np.triu(matrix, k = 1)
    masked_scores = scores + mask
    shifted_scores = masked_scores - np.max(masked_scores, axis = -1, keepdims = True)
    attention = np.exp(shifted_scores) / np.sum(np.exp(shifted_scores), axis = -1, keepdims = True)

    output = np.einsum("bld,bdk->blk", attention, V)
    return output


