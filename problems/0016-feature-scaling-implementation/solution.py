import numpy as np

def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
	mean = np.mean(data, axis = 0)
	std = np.std(data, axis = 0)
	maximum = np.max(data, axis = 0)
	minimum = np.min(data, axis = 0)

	standardized_data = (data - mean) / std
	normalized_data = (data - minimum) / (maximum - minimum)
	return standardized_data, normalized_data