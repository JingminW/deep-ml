import numpy as np

def thanksgiving_dish_predictor(preference_scores: list[float]) -> list[float]:
	"""
	Predict the probability of choosing each Thanksgiving dish using softmax.
	
	Args:
		preference_scores: List of preference scores for each dish
		(e.g., [turkey_score, stuffing_score, cranberry_score, pie_score])
		
	Returns:
		List of probabilities for each dish
	"""
	# Your code here
	preference_scores = np.array(preference_scores)
	return np.exp(preference_scores) / np.sum(np.exp(preference_scores))