import numpy as np

def lora_forward(x, W, A, B, alpha=1.0):
    x = np.asarray(x, dtype=float)
    W = np.asarray(W, dtype=float)
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    rank = A.shape[0]

    base_output = x @ W
    lora_output = (x @ B) @ A

    output = base_output + (alpha / rank) * lora_output

    return output.tolist()