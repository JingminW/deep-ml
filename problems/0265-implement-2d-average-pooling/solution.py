import numpy as np

def avg_pool_2d(input_matrix: list[list[float]], pool_size: int) -> list[list[float]]:
	"""
	Perform 2D average pooling on the input matrix.
	
	Args:
		input_matrix: 2D input array of shape (H, W)
		pool_size: Size of the square pooling window
		
	Returns:
		2D array after average pooling of shape (H//pool_size, W//pool_size)
	"""
	input_matrix = np.array(input_matrix)
	output_h, output_w = input_matrix.shape[0] // pool_size, input_matrix.shape[1] // pool_size
	output = []

	i = 0
	while i + pool_size <= input_matrix.shape[0]:
		j = 0
		while j + pool_size <= input_matrix.shape[1]:
			patch = input_matrix[i:i + pool_size, j:j + pool_size]
			output.append(np.mean(patch))

			j += pool_size

		i += pool_size

	output = np.array(output).reshape(output_h, output_w)
	return output