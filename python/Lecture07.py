import numpy as np

# Create two vectors
u = np.array([7, 3, 2, -4])
v = np.array([1, 1, -3, 2])
print("u={}, v={}".format(u, v))

# Create a matrix
A = np.matrix([[3, 4, 5, 4], [2, 2, 2, 9], [-2, 2, 7, 1], [-2, 6, 4, 4]])
print("A={}".format(A))

# Dot product between two vectors
x = u.dot(v)
print("Dot product (u.v): {}".format(x))

# Product Au
x = A.dot(u)
print("Product Au: {}".format(x))

# Product A*A
x = A.dot(A)
print("Product AA: {}".format(x))

# Transpose A^T
At = np.transpose(A)
print("A^T: {}".format(At))

# Compute determinant
detA = np.linalg.det(A)
print("Determinant of A: {}".format(detA))

# Compute inverse
Ainv = np.linalg.inv(A)
print("Inverse of A")
print(Ainv)

# Check that inverse is correct
print("A*A^-1: {}".format(A*Ainv))

get_ipython().magic('matplotlib inline')

# Set up plotting environment
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect("equal")

# Draw cube
r = [0, 1]
for s, e in combinations(np.array(list(product(r, r, r))), 2):
    if np.sum(np.abs(s - e)) == r[1] - r[0]:
        ax.plot3D(*zip(s, e), color="b", marker="o")

# Create a transformation matrix (diagonal)
A = np.array([[0.8, 0.0, 0], [0.0, 1.1, 0.0], [0.0, 0.0, 1.7]])

# Check determinant
print("Det A: {}".format(np.linalg.det(A)))

# Draw orginal cube and transformed shape
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect("equal")
r = [0, 1]
for s, e in combinations(np.array(list(product(r, r, r))), 2):
    if np.sum(np.abs(s - e)) == r[1] - r[0]:
        ax.plot3D(*zip(s, e), color="b", marker="o")
        
        s = A.dot(s)
        e = A.dot(e)
        ax.plot3D(*zip(s, e), color="r", marker="o")

# Create a transformation matrix (diagonal)
A = np.array([[0.8, 0.0, 0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.7]])

# Check determinant
print("Det A: {}".format(np.linalg.det(A)))

# Draw orginal cube and transformed shape
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect("equal")
r = [0, 1]
for s, e in combinations(np.array(list(product(r, r, r))), 2):
    if np.sum(np.abs(s - e)) == r[1] - r[0]:
        ax.plot3D(*zip(s, e), color="b", marker="o")
        s = A.dot(s)
        e = A.dot(e)
        ax.plot3D(*zip(s, e), color="r", marker="o")
        
        

# Create a transformation matrix (diagonal)
A = np.array([[0.8, 0.8, 0.8], [0.6, 1.0, 0.0], [-1.1, 0.0, 0.7]])

# Check determinant
print("Det A: {}".format(np.linalg.det(A)))

# Draw orginal cube and transformed shape
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect("equal")
r = [0, 1]
for s, e in combinations(np.array(list(product(r, r, r))), 2):
    if np.sum(np.abs(s - e)) == r[1] - r[0]:
        ax.plot3D(*zip(s, e), color="b", marker="o")
        
        s = A.dot(s)
        e = A.dot(e)
        ax.plot3D(*zip(s, e), color="r", marker="o")

# Create a transformation matrix (diagonal)
A = np.array([[2.0, 0.0, 0.0], [0.0, 1.1, 0.0], [0.0, 0.0, 1.1]])

# Check determinant
print("Det A: {}".format(np.linalg.det(A)))

# Draw orginal cube and transformed shape
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect("equal")
r = [0, 1]
for s, e in combinations(np.array(list(product(r, r, r))), 2):
    if np.sum(np.abs(s - e)) == r[1] - r[0]:
        ax.plot3D(*zip(s, e), color="b", marker="o")
        
        s = A.dot(s)
        e = A.dot(e)
        ax.plot3D(*zip(s, e), color="r", marker="o")

# Multiply A by -1 and print determinant
A = -A
print("Det A: {}".format(np.linalg.det(A)))

# Draw orginal cube and transformed shape
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect("equal")
r = [0, 1]
for s, e in combinations(np.array(list(product(r, r, r))), 2):
    if np.sum(np.abs(s - e)) == r[1] - r[0]:
        ax.plot3D(*zip(s, e), color="b", marker="o")
        
        s = A.dot(s)
        e = A.dot(e)
        ax.plot3D(*zip(s, e), color="r", marker="o")

