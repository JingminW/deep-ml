import numpy as np

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10):
    # Note: `f` is accepted only for interface parity with the PyTorch/Tinygrad
    # variants, which derive gradients from it via autograd. This version uses
    # `grad` only — the objective value `f` is never evaluated.
    x = x0
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    
    for iteration in range(1, num_iterations + 1):
        gradient = grad(x)

        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * np.power(gradient, 2)

        m_hat = m / (1 - beta1 ** iteration)
        v_hat = v / (1 - beta2 ** iteration)

        x = x - learning_rate * (m_hat / (np.sqrt((v_hat) + epsilon)))

    return x


            

        