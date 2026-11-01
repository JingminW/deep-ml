import numpy as np

def accuracy_score(y_true, y_pred):
	total = y_true.shape[0]
	return np.sum(y_true == y_pred) / total