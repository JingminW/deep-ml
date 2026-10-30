import math

def update_clusters(points, centroids, dim):
	clusters = [[] for _ in range(len(centroids))]
	for point in points:
		centroid_membership = 0
		min_dist = float('inf')

		for i, centroid in enumerate(centroids):
			if dim == 2:
				cx, cy = centroid
				x, y = point
				dist = math.sqrt((cx - x) ** 2 + (cy - y) ** 2)

			else:
				cx, cy, cz = centroid
				x, y, z = point
				dist = math.sqrt((cx - x) ** 2 + (cy - y) ** 2 + (cz - z) ** 2)

			if dist < min_dist:
				centroid_membership = i
				min_dist = dist

		clusters[centroid_membership].append(point)

	return clusters

def compute_new_centroids(clusters, dim):
	centroids = []
	for cluster in clusters:
		count, total_x, total_y, total_z = 0, 0, 0, 0
		for point in cluster:
			if dim == 2:
				x, y = point
				total_x += x
				total_y += y
				count += 1
			else:
				x, y, z = point
				total_x += x
				total_y += y
				total_z += z
				count += 1

		if dim == 2:
			cx = total_x / count
			cy = total_y / count
			centroids.append([cx, cy])
		else:
			cx = total_x / count
			cy = total_y / count
			cz = total_z / count
			centroids.append([cx, cy, cz])
	return centroids
	
def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:

	if len(points[0]) == 2:
		dim = 2
	else:
		dim = 3
	
	centroids = initial_centroids
	clusters = update_clusters(points, centroids, dim)

	for _ in range(max_iterations):
		centroids = compute_new_centroids(clusters, dim)
		clusters = update_clusters(points, centroids, dim)

	return [tuple(elem) for elem in centroids]


				
				
