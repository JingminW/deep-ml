import numpy as np
import math

def pos_encoding(position: int, d_model: int):
	if position == 0 or d_model <= 0:
		return -1

	positions = np.arange(0, position)[:, None]
	div_term = np.arange(0, d_model, 2)
	div_term = np.exp(div_term * (-math.log(10000.0) / d_model))

	pos_encoding = np.zeros((position, d_model))

	pos_encoding[:, 0::2] = np.sin(positions * div_term)
	pos_encoding[:, 1::2] = np.cos(positions * div_term)

	return pos_encoding.astype(np.float16)

	
	