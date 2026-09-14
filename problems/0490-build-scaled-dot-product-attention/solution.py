import numpy as np

def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray = None) -> tuple:
	"""
	Compute Scaled Dot-Product Attention.
	
	Args:
		Q: Query matrix of shape (seq_len_q, d_k)
		K: Key matrix of shape (seq_len_k, d_k)
		V: Value matrix of shape (seq_len_k, d_v)
		mask: Optional binary mask of shape (seq_len_q, seq_len_k)
	
	Returns:
		Tuple of (output, attention_weights)
	"""
	# Your code here
	d_k = Q.shape[1]

	attention = (Q @ K.T) / np.sqrt(d_k)
	attention -= np.max(attention, axis = 1, keepdims = True)
	if mask is not None:
		mask = np.where(mask == 0, -np.inf, 0)
		attention += mask

	softmax_sum = np.sum(np.exp(attention), axis = 1, keepdims = True)
	attention_weights = np.exp(attention) / softmax_sum
	output = attention_weights @ V

	return output, attention_weights