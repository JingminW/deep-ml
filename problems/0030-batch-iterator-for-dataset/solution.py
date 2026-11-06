import numpy as np

def batch_iterator(X, y=None, batch_size=64):
	n = len(X)

	for i in range(0, n, batch_size):
		X_batch = X[i:i + batch_size]

		if y is not None:
			y_batch = y[i:i + batch_size]
			yield [X_batch.tolist(), y_batch.tolist()]
		else:
			yield [X_batch.tolist()]
	