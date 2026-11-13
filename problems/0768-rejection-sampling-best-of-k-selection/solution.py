import numpy as np
def rejection_sampling_best_of_k(candidates, scores):
    """
    Select the highest-scoring candidate per prompt.

    Args:
        candidates: list of N lists, each containing K candidate outputs.
        scores: list of N lists, each containing K reward scores.

    Returns:
        List of N selected candidates.
    """
    # candidates = np.array(candidates)
    # scores = np.array(scores)

    # max_indices = np.argmax(scores, axis = 1)
    # return candidates[np.arange(len(candidates)), max_indices]
    result = []
    for candidate, score in zip(candidates, scores):
        max_ind = score.index(max(score))
        result.append(candidate[max_ind])

    return result