
import numpy as np


# ---------- Task 2A: Broadcasting a scalar ----------
numbers = np.array([1, 2, 3, 4, 5])

result = numbers + 5

print("Task 2A - Broadcasting result:")
print(result)


# ---------- Task 2B: Broadcasting with a 2-D matrix ----------
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])

addition = matrix + np.array([10, 20, 30])

print("\nTask 2B - Original Matrix:")
print(matrix)

print("\nMatrix addition result:")
print(addition)
