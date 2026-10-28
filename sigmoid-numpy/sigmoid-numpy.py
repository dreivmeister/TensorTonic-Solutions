import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    if isinstance(x, list):
        x = np.asarray(x)
    return 1.0 / (1 + np.exp(-x))