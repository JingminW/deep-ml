import numpy as np

def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
	h1 = np.maximum(0, w1 @ x)
	h2 = np.maximum(0, x + w2 @ h1)
	return h2