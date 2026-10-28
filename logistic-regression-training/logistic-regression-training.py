import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def model(X, w, b):
    z = X @ w + b
    return _sigmoid(z)

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    samples, feat = X.shape
    w = np.zeros(feat)
    b = np.zeros(1)

    for i in range(steps):
        p = model(X, w, b)
        dLdw = 1 / samples * X.T @ (p - y)
        dldb = 1 / samples * sum(p - y)

        w -= lr * dLdw
        b -= lr * dldb
    return (w, b)