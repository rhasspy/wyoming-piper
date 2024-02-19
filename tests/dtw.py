import math

import numpy as np
import scipy


def compute_optimal_path(x: np.ndarray, y: np.ndarray) -> float:
    """Computes optimal path between x and y."""
    m = len(x)
    n = len(y)

    # Need 2-D arrays for distance calculation
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)

    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    distance_matrix = scipy.spatial.distance.cdist(x, y, metric="cosine")

    cost_matrix = np.full(shape=(m, n), fill_value=math.inf, dtype=float)
    cost_matrix[0][0] = distance_matrix[0][0]

    for row in range(1, m):
        cost = distance_matrix[row, 0]
        cost_matrix[row][0] = cost + cost_matrix[row - 1][0]

    for col in range(1, n):
        cost = distance_matrix[0, col]
        cost_matrix[0][col] = cost + cost_matrix[0][col - 1]

    for row in range(1, m):
        for col in range(1, n):
            cost = distance_matrix[row, col]
            cost_matrix[row][col] = cost + min(
                cost_matrix[row - 1][col],  # insertion
                cost_matrix[row][col - 1],  # deletion
                cost_matrix[row - 1][col - 1],  # match
            )

    distance = cost_matrix[m - 1][n - 1]

    return distance
