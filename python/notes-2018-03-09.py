import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from numpy.linalg import matrix_power as mpow
get_ipython().magic('matplotlib inline')

def add_row(A,k,i,j):
    "Add k times row i to row j in matrix A (using 0 indexing)."
    m = A.shape[0] # The number of rows of A
    E = np.eye(m)
    if i != j:
        E[j,i] = k
    else:
        E[i,j] = k+1
    return E@A

def swap_row(A,i,j):
    "Swap rows i and j in matrix A (using 0 indexing)."
    nrows = A.shape[0] # The number of rows in A
    E = np.eye(nrows)
    E[i,i] = 0
    E[j,j] = 0
    E[i,j] = 1
    E[j,i] = 1
    return E@A

def scale_row(A,k,i):
    "Multiply row i by k in matrix (using 0 indexing)."
    nrows = A.shape[0] # The number of rows in A
    E = np.eye(nrows)
    E[i,i] = k
    return E@A

A = np.array([[6,15,1],[8,7,12],[2,7,8]])
b = np.array([[2],[14],[10]])

print(A)

M = np.hstack([A,b])
print(M)

M1 = scale_row(M,1/6,0)
print(M1)

M2 = add_row(M1,-8,0,1)
print(M2)

M3 = add_row(M2,-2,0,2)
print(M3)

M4 = scale_row(M3,-1/13,1)
print(M4)

M5 = add_row(M4,-2,1,2)
print(M5)

M6 = scale_row(M5,1/M5[2,2],2)
print(M6)

M7 = add_row(M6,-M6[1,2],2,1)
print(M7)

M8 = add_row(M7,-M7[0,2],2,0)
print(M8)

M9 = add_row(M8,-M8[0,1],1,0)
print(M9)

x = M9[:,3]
print(x)

x = la.solve(A,b)
print(x)

print(A)

la.det(A)

la.inv(A)

Ainv = la.inv(A)
x = Ainv @ b
print(x)

A @ Ainv

A.T

np.transpose(A)

np.trace(A)

6+7+8

S = A @ A.T

print(S)

evals, evecs = la.eig(S)

print(evals)

evals.dtype

evals = evals.real

print(evals)

evecs

(1/evals[0]) * S @ evecs[:,0]

np.dot(evecs[:,0],evecs[:,1])

