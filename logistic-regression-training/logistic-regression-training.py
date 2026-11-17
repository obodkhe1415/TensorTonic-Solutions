
import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.where(
        z >= 0,
        1 / (1 + np.exp(-z)),
        np.exp(z) / (1 + np.exp(z))
    )

def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    steps: int = 1000
) -> tuple[np.ndarray, float]:

    m, n = X.shape

    # Initialize weights and bias
    w = np.zeros(n)
    b = 0.0

    # Training loop
    for _ in range(steps):

        # Calculate predictions
        z = np.dot(X, w) + b
        y_pred = _sigmoid(z)

        # Calculate error
        error = y_pred - y

        # Calculate gradients
        dw = np.dot(X.T, error) / m
        db = np.sum(error) / m

        # Update weights and bias
        w -= lr * dw
        b -= lr * db

    return w, b
