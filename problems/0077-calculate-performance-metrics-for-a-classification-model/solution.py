import numpy as np
def performance_metrics(actual: list[int], predicted: list[int]) -> tuple:
	actual = np.array(actual)
	predicted = np.array(predicted)

	accuracy = np.sum(actual == predicted) / len(actual)
	pos_inds = np.where(actual == 1)[0]
	neg_inds = np.where(actual == 0)[0]

	true_positives = np.sum(predicted[pos_inds])
	false_positives = np.sum(predicted[neg_inds])
	true_negatives = np.sum(1 - predicted[neg_inds])
	false_negatives = np.sum(1 - predicted[pos_inds])

	precision = true_positives / (true_positives + false_positives)
	recall = true_positives / (true_positives + false_negatives)
	f1 = (2 * precision * recall) / (precision + recall)
	specificity = true_negatives / (true_negatives + false_positives)
	negativePredictive = true_negatives / (true_negatives + false_negatives)

	confusion_matrix = [[true_positives, false_negatives], [false_positives, true_negatives]]

	return confusion_matrix, round(accuracy, 3), round(f1, 3), round(specificity, 3), round(negativePredictive, 3)
