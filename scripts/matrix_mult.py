import numpy as np

m1 = np.array([[1, 5, 8, 6],
               [1, 3, 7, 6],
               [1, 3, 4, 6],
               [1, 2, 7, 6]])
m2 = np.array([[1, 3, 8, 6],
               [1, 3, 9, 6],
               [1, 1, 1, 1],
               [1, 2, 7, 6]])

print([j for i in m1.tolist() for j in i], [j for i in m2.tolist() for j in i])

print([j for i in (m1 @ m2).tolist() for j in i])

# [20, 38, 103, 80, 17, 31, 84, 67, 14, 28, 81, 64, 16, 28, 75, 61]
