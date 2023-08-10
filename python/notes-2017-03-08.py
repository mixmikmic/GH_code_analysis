import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
get_ipython().magic('matplotlib inline')

M = np.random.randint(0,10,[3,3])
M

M @ M

M.T

M @ M.T

A = np.array([[1,2],[3,4]])
A

la.inv(A)

la.det(A)

np.trace(A)

A

trace_A = np.trace(A)
det_A = la.det(A)
I = np.eye(2)

A @ A - trace_A * A + det_A * I

N = np.random.randint(0,10,[2,2])
N

trace_N = np.trace(N)
det_N = la.det(N)
I = np.eye(2)
N @ N - trace_N * N + det_N * I

n = 4
P = np.random.randint(0,10,[n,n])
P

S = P @ P.T # This is a symmetric matrix
S

get_ipython().magic('pinfo la.eig')

evals, evecs = la.eig(S)

evals

evecs

v1 = evecs[:,0] # First column is the first eigenvector
v1

v2 = evecs[:,1] # Second column is the second eigenvector
v2

v1 @ v2

