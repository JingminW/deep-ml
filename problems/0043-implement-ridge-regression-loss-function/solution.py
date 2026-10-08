import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
	y_pred = X @ w
	mse = np.mean((y_pred - y_true) ** 2)
	l2_term = np.linalg.norm(w) ** 2 * alpha
	return mse + l2_term
