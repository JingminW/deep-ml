import numpy as np

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray):
    d_k = Q.shape[1]

    attention = (Q @ K.T) / np.sqrt(d_k)
    attention = attention - np.max(attention, axis = 1, keepdims = True)
    attention = np.exp(attention) / np.sum(np.exp(attention), axis = 1, keepdims = True)
    output = attention @ V

    return output

def multihead_self_attention(Q_list: list, K: np.ndarray, V: np.ndarray, n_heads):
    attention_outputs = []

    for i in range(n_heads):
        attention_outputs.append(self_attention(Q_list[i], K, V))

    return np.concatenate(attention_outputs, axis = 1)


def multiquery_attention(X: np.ndarray, W_queries: list, W_key: np.ndarray, W_value: np.ndarray, W_out: np.ndarray) -> np.ndarray:
    """
    Compute Multi-Query Attention.
    
    Args:
        X: Input array of shape (seq_len, d_model)
        W_queries: List of query weight matrices, each (d_model, d_k), one per head
        W_key: Shared key weight matrix of shape (d_model, d_k)
        W_value: Shared value weight matrix of shape (d_model, d_v)
        W_out: Output projection matrix of shape (num_heads * d_v, d_model)
    
    Returns:
        Output array of shape (seq_len, d_model), rounded to 4 decimal places
    """
    n_heads = len(W_queries)
    queries = []
    for i in range(n_heads):
        queries.append(X @ W_queries[i])

    K = X @ W_key
    V = X @ W_value

    attention = multihead_self_attention(queries, K, V, n_heads)
    output = attention @ W_out
    return output

    

    