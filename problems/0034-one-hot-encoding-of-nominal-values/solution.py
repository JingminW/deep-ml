import numpy as np

def to_categorical(x, n_col=None):
	if n_col:
		one_hot = np.eye(n_col)
	else:
		num_classes = len(np.unique(x))
		one_hot = np.eye(num_classes)

	return one_hot[x]