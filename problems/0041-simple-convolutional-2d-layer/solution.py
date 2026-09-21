import numpy as np
import math

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
	input_height, input_width = input_matrix.shape
	kernel_height, kernel_width = kernel.shape

	input_matrix = np.pad(input_matrix, pad_width = padding, mode = 'constant', constant_values = 0)
	new_input_height, new_input_width = input_matrix.shape

	row = 0
	output = []
	while row + kernel_height <= new_input_height:
		col = 0
		while col + kernel_width <= new_input_width:
			image_patch = input_matrix[row:row + kernel_height, col:col + kernel_width]
			output.append(np.sum(image_patch * kernel))

			col += stride

		row += stride
	
	output_height = math.floor((input_height + 2 * padding - kernel_height) / stride) + 1
	output_width = math.floor((input_width + 2 * padding - kernel_width) / stride) + 1
	output_matrix = np.array(output).reshape(output_height, output_width)

	return output_matrix
