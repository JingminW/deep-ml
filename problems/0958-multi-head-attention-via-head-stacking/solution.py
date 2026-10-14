import numpy as np

def multi_head_attention_stack(x, heads):
    """
    Multi-head causal self-attention by stacking single-head attention modules.

    Args:
        x: array-like of shape (B, T, d_in)
        heads: list of H tuples (W_q, W_k, W_v); each weight is shape (d_in, d_head)

    Returns:
        numpy.ndarray of shape (B, T, H * d_head)
    """
    x = np.asarray(x)
    B, T, _ = x.shape

    head_outputs = []

    for W_q, W_k, W_v in heads:
        W_q = np.asarray(W_q)
        W_k = np.asarray(W_k)
        W_v = np.asarray(W_v)

        # (B, T, d_in) @ (d_in, d_head)
        # -> (B, T, d_head)
        Q = np.einsum("btd,dh->bth", x, W_q)
        K = np.einsum("btd,dh->bth", x, W_k)
        V = np.einsum("btd,dh->bth", x, W_v)

        d_head = Q.shape[-1]
        scores = np.einsum("bqd,bkd->bqk", Q, K) / np.sqrt(d_head)

        causal_mask = np.triu(np.full((T, T), -np.inf), k = 1)
        scores = scores + causal_mask

        scores = scores - np.max(scores, axis = -1, keepdims = True)

        exp_scores = np.exp(scores)
        attention = exp_scores / np.sum(
            exp_scores,
            axis = -1,
            keepdims=True
        )
        context = np.einsum("bqk,bkd->bqd", attention, V)
        head_outputs.append(context)

    return np.concatenate(head_outputs, axis=-1)
    
