import numpy as np
from numpy.linalg import inv, norm
import matplotlib.pyplot as plt

def gauss_seidel(A,b,tol):
    D = np.diag(np.diag(A))
    L = (np.tril(A)-D)
    U = (np.triu(A)-D)

    err = 1
    errs = list()
    step = 0
    x_old = np.zeros((len(b),1))
    while err > 10**-6:
        x = np.dot(inv(inv(D).dot(L)+np.identity(len(b))),inv(D).dot(b - U.dot(x_old)))
        err = norm(b-A.dot(x), 2)
        errs.append(err)
        x_old = x
        step += 1
    return [x,step,errs]



