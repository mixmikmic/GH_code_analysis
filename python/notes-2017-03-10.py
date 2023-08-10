import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

M = np.array([[2,1,],[-2,5]])
M

M @ M

M**2

np.linalg.matrix_power(M,2)

from numpy.linalg import matrix_power as mpow

mpow(M,2)

mpow(M,5)

P = np.array([[1,1],[1,-1]])
P

D = np.diag((3,1))
D

M = P @ D @ la.inv(P)
M

evals, evecs = la.eig(M)

evals

evecs

Pinv = la.inv(P)

k = 20

get_ipython().magic('timeit mpow(M,k)')

get_ipython().magic('timeit P @ D**k @ Pinv')

def proj(v,w):
    '''Project vector v onto w.'''
    v = np.array(v)
    w = np.array(w)
    return np.sum(v * w)/np.sum(w * w) * w   # or (v @ w)/(w @ w) * w

proj([1,2,3],[1,1,1])

