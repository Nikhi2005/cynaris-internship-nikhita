import numpy as np

# Task 4
# Matrix Multiplication

A = np.array([
    [1, 2],
    [3, 4]
])

B = np.array([
    [5, 6],
    [7, 8]
])

result = A @ B

print("Matrix A:\n", A)
print("Matrix B:\n", B)
print("Matrix multiplication result:\n", result)