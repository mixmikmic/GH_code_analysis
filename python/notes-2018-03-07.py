import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import scipy.linalg as la

A = np.array([[1,2],[3,4]])
print(A)

A*A

A @ A

A @ A @ A

from numpy.linalg import matrix_power as mpow

mpow(A,3)

D = np.array([[2,0],[0,-1]])
print(D)

mpow(D,5)

get_ipython().magic('pinfo la.solve')

A = np.random.randint(-5,5,(3,3))
print(A)

b = np.random.randint(-5,5,(3,1))
print(b)

x = la.solve(A,b)
print(x)

A @ x

def add_row(A,k,i,j):
    "Add k times row i to row j in matrix A (using 0 indexing)."
    m = A.shape[0] # The number of rows of A
    E = np.eye(m)
    E[j,i] = k
    return E@A

M = np.array([[1,1],[3,2]])
add_row(M,2,0,1)

def swap_row(A,i,j):
    "Swap rows i and j in matrix A (using 0 indexing)."
    nrows = A.shape[0] # The number of rows in A
    E = np.eye(nrows)
    E[i,i] = 0
    E[j,j] = 0
    E[i,j] = 1
    E[j,i] = 1
    return E@A

M = np.array([[1,1],[3,2]])
swap_row(M,0,1)

def scale_row(A,k,i):
    "Multiply row i by k in matrix (using 0 indexing)."
    nrows = A.shape[0] # The number of rows in A
    E = np.eye(nrows)
    E[i,i] = k
    return E@A

M = np.array([[1,1],[3,2]])
scale_row(M,3,1)

M

b = np.array([[1],[-1]])
print(b)

A = np.hstack([M,b])
print(A)

A1 = add_row(A,-3,0,1)
print(A1)

A2 = scale_row(A1,-1,1)
print(A2)

A3 = add_row(A2,-1,1,0)
print(A3)

x = la.solve(M,b)
print(x)

