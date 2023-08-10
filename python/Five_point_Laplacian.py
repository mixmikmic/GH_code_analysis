get_ipython().magic('matplotlib inline')
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt

def five_pt_laplacian_sparse(m,a,b):
    """Construct a sparse finite difference matrix that approximates the Laplacian."""
    e=np.ones(m**2)
    e2=([1]*(m-1)+[0])*m
    e3=([0]+[1]*(m-1))*m
    h=(b-a)/(m+1)
    A=scipy.sparse.spdiags([-4*e,e2,e3,e,e],[0,-1,1,-m,m],m**2,m**2)
    A/=h**2
    A = A.todia()
    return A

A = five_pt_laplacian_sparse(4,-1.,1.)
plt.spy(A)

