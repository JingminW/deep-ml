import numpy as np

class LSTM:
	def __init__(self, input_size, hidden_size):
		self.input_size = input_size
		self.hidden_size = hidden_size

		# Initialize weights and biases
		self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wo = np.random.randn(hidden_size, input_size + hidden_size)

		self.bf = np.zeros((hidden_size, 1))
		self.bi = np.zeros((hidden_size, 1))
		self.bc = np.zeros((hidden_size, 1))
		self.bo = np.zeros((hidden_size, 1))

	def forward(self, x, initial_hidden_state, initial_cell_state):
		"""
		Processes a sequence of inputs and returns the hidden states, final hidden state, and final cell state.
		"""
		seq_len, input_size = x.shape
		hidden_state = np.empty((self.hidden_size))
		cell_state = np.empty((self.hidden_size))
		hidden_states = [initial_hidden_state]

		for i in range(seq_len):
			if i == 0:
				prev_cell_state = initial_cell_state
				prev_hidden_state = initial_hidden_state
			combined = np.concatenate([prev_hidden_state, x[i][:, None]], axis=0)

			forget_gate = self.sigmoid(self.Wf @ combined + self.bf)
			input_gate = self.sigmoid(self.Wi @ combined + self.bi)
			output_gate = self.sigmoid(self.Wo @ combined + self.bo)
			candidate_cell_state = np.tanh(self.Wc @ combined + self.bc)

			cell_state = forget_gate * prev_cell_state + input_gate * candidate_cell_state
			hidden_state = output_gate * np.tanh(cell_state)
			hidden_states.append(hidden_state)

			prev_cell_state = cell_state
			prev_hidden_state = hidden_state

		return hidden_states, hidden_state, cell_state

	def sigmoid(self, z):
		return 1 / (1 + np.exp(-z))
