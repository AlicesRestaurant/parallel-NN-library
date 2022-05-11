import numpy as np

m1 = np.array([[1,5,8,6],
               [1,3,7,6],
               [1,3,4,6],
               [1,2,7,6]])
m2 = np.array([[1,3,8,6],
               [1,3,9,6],
               [1,1,1,1],
               [1,2,7,6]])

print([j for i in m1.tolist() for j in i], [j for i in m2.tolist() for j in i])

print(m1 @ m2)