
import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """

    # Convert input list into NumPy array
    A = np.array(A)

    # Get number of rows and columns
    rows, cols = A.shape

    # Create a new matrix with reversed dimensions
    result = np.zeros((cols, rows), dtype=A.dtype)

    # Swap rows and columns using indexing
    for i in range(rows):
        for j in range(cols):
            result[j][i] = A[i][j]

    return result
