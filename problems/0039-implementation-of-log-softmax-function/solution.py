import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	return np.log(np.exp(scores) / np.sum(np.exp(scores)))