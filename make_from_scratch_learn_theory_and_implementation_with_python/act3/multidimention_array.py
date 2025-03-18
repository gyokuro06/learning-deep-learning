# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
# ]
# ///

import numpy as np

A = np.array([1, 2, 3, 4])
print(A)
print(np.ndim(A))
print(A.shape)
print(A.shape[0])
print()

B = np.array([[1, 2], [3, 4], [5, 6]])
print(B)
print(np.ndim(B))
print(B.shape)
print()

C = np.array([[1, 2], [3, 4]])
D = np.array([[5, 6], [7, 8]])
print(C.shape)
print(D.shape)
print("C・D")
print(np.dot(C, D))
print("D・C")
print(np.dot(D, C))
