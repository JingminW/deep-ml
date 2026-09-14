import numpy as np

def rnn_forward(input_sequence: list[list[float]], initial_hidden_state: list[float], Wx: list[list[float]], Wh: list[list[float]], b: list[float]) -> list[float]:
	input_sequence = np.array(input_sequence)
	initial_hidden_state = np.array(initial_hidden_state)
	Wx = np.array(Wx)
	Wh = np.array(Wh)
	b = np.array(b)

	hidden_state = initial_hidden_state[:, None]

	for sequence in input_sequence:
		hidden_state = Wx @ sequence[:, None] + Wh @ hidden_state + b[:, None]

		hidden_state = np.tanh(hidden_state)

	final_hidden_state = hidden_state 
	return final_hidden_state.reshape(-1)