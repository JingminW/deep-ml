import numpy as np

class DropoutLayer:
    def __init__(self, p: float):
        """Initialize the dropout layer.
        
        Attributes to set:
            self.p: the dropout rate
            self.mask: stores the dropout mask (initially None)
        """
        self.p = p
        self.mask = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass of the dropout layer.
        
        Generate a new mask on each training forward pass and store it in self.mask.
        """
        # rng = np.random.default_rng()

        if training:
            mask = np.random.binomial(n = 1, p = 1 - self.p, size = x.shape)
            self.mask = mask
            masked_output = mask * x
            scaled_output = (1 / (1 - self.p)) * masked_output
            return scaled_output
        else:
            return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass of the dropout layer.
        
        Use the stored self.mask from the most recent forward pass.
        """
        masked_grad = self.mask * grad
        scaled_grad = (1 / (1 - self.p)) * masked_grad
        return scaled_grad