import numpy as np
from numpy.linalg import inv, norm
import matplotlib.pyplot as plt

def sor(A,b,tol,w):
    D = np.diag(np.diag(A))
    L = (np.tril(A)-D)
    U = (np.triu(A)-D)
    err = 1
    errs = list()
    step = 0
    x_old = np.zeros((len(b),1))
    while err > 10**-6:
        x = inv(w*L + D).dot((1-w)*D.dot(x_old)-w*U.dot(x_old)) + w*inv(D+w*L).dot(b)
        err = norm(b-A.dot(x), 2)
        errs.append(err)
        x_old = x
        step += 1
    
    return [x,step,errs]



