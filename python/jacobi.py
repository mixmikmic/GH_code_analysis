import numpy as np
from numpy.linalg import inv, norm
import matplotlib.pyplot as plt

#Jacobi Method
# Input: matrix A, vector b, and tolerance
# Output: solution x, number of steps, list of errors using backward error
def jacobi(A,b,tol):
    D = np.diag(np.diag(A))
    U = (np.triu(A)-D)
    L = (np.tril(A)-D)
    
    err = 1
    errs = list()
    step = 0
    x_old = np.zeros((len(b),1))
    while err > tol:
        x = inv(D).dot(b -(L+U).dot(x_old))
        err = norm(b-A.dot(x), 2)
        errs.append(err)
        x_old = x
        step += 1
    return [x,step,errs]



