import numpy as np

def shuffle_data(X, y, seed=None):
	np.random.seed(seed)
	n = X.shape[0]
	shuffled_idxs = np.random.permutation(n)
	X_shuffled = X[shuffled_idxs]
	y_shuffled = y[shuffled_idxs]

	return X_shuffled, y_shuffled