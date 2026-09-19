import torch
import torch.nn.functional as F


def grouped_query_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, num_heads: int, num_kv_heads: int) -> torch.Tensor:
    """
    Compute Grouped Query Attention.

    Args:
        Q: Query tensor, shape (batch_size, seq_len, num_heads * head_dim)
        K: Key tensor, shape (batch_size, seq_len, num_kv_heads * head_dim)
        V: Value tensor, shape (batch_size, seq_len, num_kv_heads * head_dim)
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads

    Returns:
        Output tensor, shape (batch_size, seq_len, num_heads * head_dim)
    """
    batch_size, seq_len, q_dim = Q.shape

    if num_heads % num_kv_heads != 0:
        raise ValueError(
            "num_heads must be evenly divisible by num_kv_heads"
        )

    if q_dim % num_heads != 0:
        raise ValueError(
            "Q.shape[-1] must be divisible by num_heads"
        )

    head_dim = q_dim // num_heads
    heads_per_group = num_heads // num_kv_heads

    expected_kv_dim = num_kv_heads * head_dim

    # Q:
    # (b, l, num_heads * head_dim)
    # ->
    # (b, l, num_kv_heads, heads_per_group, head_dim)
    Q = Q.reshape(
        batch_size,
        seq_len,
        num_kv_heads,
        heads_per_group,
        head_dim
    )

    # K, V:
    # (b, l, num_kv_heads * head_dim)
    # ->
    # (b, l, num_kv_heads, head_dim)
    K = K.reshape(
        batch_size,
        seq_len,
        num_kv_heads,
        head_dim
    )

    V = V.reshape(
        batch_size,
        seq_len,
        num_kv_heads,
        head_dim
    )

    # Q: (b, q, g, h, d)
    # K: (b, k, g, d)
    #
    # Output:
    # scores: (b, q, g, h, k)
    scores = torch.einsum(
        "bqghd,bkgd->bqghk",
        Q,
        K
    )

    scores = scores / (head_dim ** 0.5)

    # Softmax across key positions
    attention_weights = F.softmax(
        scores,
        dim=-1
    )

    # attention_weights: (b, q, g, h, k)
    # V:                 (b, k, g, d)
    #
    # output:            (b, q, g, h, d)
    output = torch.einsum(
        "bqghk,bkgd->bqghd",
        attention_weights,
        V
    )

    # Merge group/head dimensions back together
    output = output.reshape(
        batch_size,
        seq_len,
        num_heads * head_dim
    )

    return output
