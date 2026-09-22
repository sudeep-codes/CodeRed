# pyrefly: ignore [missing-import]
import numpy as np

CLASSES = ["RED", "YELLOW", "GREEN", "BLACK"]

# rows = actual, columns = predicted — copy exactly from the problem statement
COST_MATRIX = np.array([
    [0, 5, 10, 5],   # actual RED
    [2, 0, 3,  2],   # actual YELLOW
    [1, 1, 0,  1],   # actual GREEN
    [2, 2, 2,  0],   # actual BLACK
])

def total_cost(y_true, y_pred):
    """y_true, y_pred: arrays of class labels (strings from CLASSES)."""
    idx = {c: i for i, c in enumerate(CLASSES)}
    cost = 0
    for t, p in zip(y_true, y_pred):
        cost += COST_MATRIX[idx[t], idx[p]]
    return cost

