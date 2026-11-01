import numpy as np

def divide_on_feature(X, feature_i, threshold):
	mask = X[:, feature_i] >= threshold
	greater_equal = X[mask]
	less = X[~mask]

	return [greater_equal, less]