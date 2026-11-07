import numpy as np
def precision(y_true, y_pred):
	pos_inds = np.where(y_true == 1)[0]
	neg_inds = np.where(y_true == 0)[0]

	true_positives = np.sum(y_pred[pos_inds])
	false_positives = np.sum(y_pred[neg_inds])

	return true_positives / (true_positives + false_positives) if true_positives > 0.0 else 0.0




