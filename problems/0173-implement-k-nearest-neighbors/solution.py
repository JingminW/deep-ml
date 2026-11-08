import numpy as np

def k_nearest_neighbors(points, query_point, k):
    """
    Find k nearest neighbors to a query point
    
    Args:
        points: List of tuples representing points [(x1, y1), (x2, y2), ...]
        query_point: Tuple representing query point (x, y)
        k: Number of nearest neighbors to return
    
    Returns:
        List of k nearest neighbor points as tuples
        When distances are tied, points appearing earlier in the input list come first.
    """
    points = np.asarray(points)
    query_point = np.asarray(query_point)

    distances = np.linalg.norm(points - query_point, axis = 1)
    sorted_inds = np.argsort(distances)[:k]
    top_k = points[sorted_inds]

    return [tuple(point) for point in top_k]
    