import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    m, n = X.shape
    for _ in range(n_epochs):
        if method == "batch":
            y_pred = X @ weights
            grad = (2 / m) * (X.T @ (y_pred - y))
            weights = weights - learning_rate * grad

        elif method == "stochastic":
            for i in range(m):
                y_pred = X[i] @ weights
                grad = 2 * X[i] * (y_pred - y[i])
                weights = weights - learning_rate * grad

        else:
            for i in range(0, m , batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]

                k = len(X_batch)

                y_pred = X_batch @ weights
                grad = (2 / k) * (X_batch.T @ (y_pred - y_batch))

                weights = weights - learning_rate * grad

    return weights

