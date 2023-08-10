import numpy as np
import numpy.linalg as la

def newton_nn(F, JF, x0, nn_ind, tol=1e-4, maxiter=5):
    """Newton's method for solving a system of equations in higher dimensions, ensuring the entries stay non-negative.
    This implementation records and returns the whole trajectory.
    F:       Equations to be solved
    JF:      Jacobian matrix
    nn_ind:  indices in which to ensure non-negativity
    tol:     stop if norm of difference of iterates below tol
    maxiter: maximum iterations (does not need to be large for Newton's method)
    """
    n = x0.shape[0]
    x = np.zeros((n,maxiter+1))
    # Initialize 0-th iterate to some big number, and first one to x0
    x[:,1] = x0
    x[:,0] = x0+10*tol*np.ones(n)
    i = 1
    
    while la.norm(x[:,i]-x[:,i-1])>tol and i<maxiter:
        delta = la.solve(JF(x[:,i]), F(x[:,i]))
        # find optimal steplength
        alpha = 1.
        ind = np.argmin(x[nn_ind,i]-delta[nn_ind])
        if x[ind,i]-delta[ind]<0:
            alpha = x[ind,i]/delta[i]
        xnew = x[:,i]-alpha*delta
        x[:,i+1] = xnew
        i += 1
    return x[:,1:i+2]

def F(x):
    """The x variables are x[0] to x[2], the y variables x[3] to x[4], and s are x[5] to x[7]
    """
    return np.array([x[3]+x[5]-1, 
                    x[4]-x[6]-2,
                    -2*x[3]-x[4]+x[7]+2,
                    x[0]-2*x[2]-1,
                    x[1]-x[2]+1,
                    x[0]*x[5],
                    x[1]*x[6],
                    x[2]*x[7]])

def JF(x):
    return np.array([[0, 0, 0,  1,  0, 1, 0, 0],
                     [0, 0, 0,  0,  1, 0, 1, 0],
                     [0, 0, 0, -2, -1, 0, 0, 1],
                     [1, 0, -2, 0,  0, 0, 0, 0],
                     [0, 1, -1, 0,  0, 0, 0, 0],
                     [x[5], 0, 0, 0, 0, x[0], 0, 0],
                     [0, x[6], 0, 0, 0, 0, x[1], 0],
                     [0, 0, x[7], 0, 0, 0, 0, x[2]]])

nn_ind = [0, 1, 2, 5, 6, 7]

x0 = np.ones(8)
x0[3:5] = np.array([0.8,1])
xout = newton_nn(F, JF, x0, nn_ind)

xout[0:3,:5]



