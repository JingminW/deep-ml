import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gru_cell(x: np.ndarray, h_prev: np.ndarray,
             W_z: np.ndarray, U_z: np.ndarray, b_z: np.ndarray,
             W_r: np.ndarray, U_r: np.ndarray, b_r: np.ndarray,
             W_h: np.ndarray, U_h: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """
    Implements a single GRU cell forward pass.
    
    Args:
        x: Input vector of shape (input_size,)
        h_prev: Previous hidden state of shape (hidden_size,)
        W_z, W_r, W_h: Weight matrices for input
        U_z, U_r, U_h: Weight matrices for hidden state
        b_z, b_r, b_h: Bias vectors
    
    Returns:
        h_next: New hidden state of shape (hidden_size,)
    """
    reset_gate = sigmoid(W_r @ x + U_r @ h_prev + b_r)
    update_gate = sigmoid(W_z @ x + U_z @ h_prev + b_z)

    candidate_hidden = np.tanh(W_h @ x + U_h @ (reset_gate * h_prev) + b_h)
    hidden_state = (1 - update_gate) * h_prev + update_gate * candidate_hidden
    return hidden_state