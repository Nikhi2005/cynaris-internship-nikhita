
import numpy as np

# ---------- 1-D array ----------
array_1d = np.array([1, 2, 3, 4, 5])

print("One-dimensional Array")
print("1D array:\n", array_1d)
print(f"Array shape = {array_1d.shape}\n")


# ---------- 2-D array ----------
array_2d = np.array([[1, 2, 3],
                     [4, 5, 6]])

print("Two-dimensional Array")
print("2D array:\n", array_2d)
print(f"Array shape = {array_2d.shape}\n")


# ---------- 3-D array ----------
array_3d = np.array([[[1, 2], [3, 4]],
                     [[5, 6], [7, 8]]])

print("Three-dimensional Array")
print("3D array:\n", array_3d)
print(f"Array shape = {array_3d.shape}")