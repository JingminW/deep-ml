import numpy as np

def layer_norm(x, gamma, beta, eps):
	x_mean = np.mean(x, axis = -1, keepdims = True)
	x_var = np.var(x, axis = -1, keepdims = True)

	x_norm = (x - x_mean) / np.sqrt(x_var + eps)
	return gamma * x_norm + beta

def forward(z, W, b):
	return z @ W + b

def transformer_block(x: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray, gamma2: np.ndarray, beta2: np.ndarray, mode: str, eps: float = 1e-5) -> np.ndarray:
	"""
	Apply a transformer block with two sublayers using either pre-norm or post-norm.
	
	Args:
		x: Input array of shape (seq_len, d_model)
		W1, b1: Weights and bias for first sublayer
		W2, b2: Weights and bias for second sublayer
		gamma1, beta1: LayerNorm params for first normalization
		gamma2, beta2: LayerNorm params for second normalization
		mode: 'pre_norm' or 'post_norm'
		eps: Epsilon for numerical stability
	
	Returns:
		Output array of shape (seq_len, d_model)
	"""
	if mode not in ["pre_norm", "post_norm"]:
		raise ValueError("Invalid layer norm mode")

	if mode == "pre_norm":
		h = x + forward(layer_norm(x, gamma1, beta1, eps), W1, b1)
		y = h + forward(layer_norm(h, gamma2, beta2, eps), W2, b2)
	else:
		h = layer_norm(x + forward(x, W1, b1), gamma1, beta1, eps)
		y = layer_norm(h + forward(h, W2, b2), gamma2, beta2, eps)

	return y

