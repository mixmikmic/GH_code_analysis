import numpy as np
from numpy.linalg import inv, pinv, eigvals
np.set_printoptions(precision=3, suppress=True)
from IPython.display import display

def multiplication_matrix(d, xs, pis = None):
    """
    Construct a d x d multiplication matrix as shown above, 
    using the values xs = (x_1, ..., x_k) and pis = (pi_1, ... pi_k).
    """
    # Initialize pis appropriately
    if pis is not None:
        assert len(pis) == len(xs)
    else:
        pis = np.ones(len(xs))/len(xs)
    
    M = np.zeros((d,d))
    for i in xrange(d):
        for j in xrange(d):
            for pi, x in zip(pis, xs):
                M[i,j] += pi * (x**(i+j))
    return M

def multiplication_matrix_plus(d, xs, pis = None):
    """
    Construct a d x d multiplication matrix as shown above, 
    using the values xs = (x_1, ..., x_k) and pis = (pi_1, ... pi_k).
    """
    # Initialize pis appropriately
    if pis is not None:
        assert len(pis) == len(xs)
    else:
        pis = np.ones(len(xs))/len(xs)
    
    M = np.zeros((d,d))
    for i in xrange(d):
        for j in xrange(d):
            for pi, x in zip(pis, xs):
                M[i,j] += pi * (x**(i+j+1))
    return M

d, xs, pis = 3, [1, 2, 3], None
M = multiplication_matrix(d, xs, pis)
M_ = multiplication_matrix_plus(d, xs, pis)
display(M)
display(M_)
C = pinv(M).dot(M_)
display(C)
xs_ = eigvals(C)
display(xs_)

d, xs, pis = 5, [-0.5, .25, .9], None
M = multiplication_matrix(d, xs, pis)
M_ = multiplication_matrix_plus(d, xs, pis)
display(M)
display(M_)
C = pinv(M).dot(M_)
display(C)
xs_ = eigvals(C)
display(xs_)

import scipy
import scipy.linalg
from util import orthogonal
k = 3
U = orthogonal(d)[:k,:]
N = U.dot(M).dot(U.T)
N_ = U.dot(M_).dot(U.T)
display(N)
display(N_)
D = inv(N).dot(N_)
display(D)
ys_ = eigvals(D)
display(ys_)

