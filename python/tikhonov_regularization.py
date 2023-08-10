get_ipython().magic('matplotlib inline')


import numpy as np
from numpy import dot
from numpy.linalg import solve
import matplotlib.pyplot as plt

sigma = 0.5   # Parameterizes measurement error for c
lmda = 1      # Regularization parameter
numreps = 10  # Number of times to solve system

# Construct an arbitrary N x K matrix A 
N, K = 40, 20
A = np.empty((N, K))
A[:, 0] = 1
for i in range(1, K):
    A[:, i] = A[:, i-1] + 0.1 * np.random.randn(N)

I = np.identity(K)             # K x K identity
Ap = A.T                       # A transpose
bstar = np.zeros((K, 1)) + 10  # True solution
c = dot(A, bstar)              # Corresponding c
index = range(1, K+1)          # For plotting

fig, ax = plt.subplots(figsize=(12, 8))

bbox = (0., 1.02, 1., .102)
legend_args = {'bbox_to_anchor': bbox, 'loc': 3, 'mode': 'expand'}

# Plot the solutions
for j in range(numreps):
    # Observe c with error
    c0 = c.flatten() + sigma * np.random.randn(N)
    # Compute the regularized solution
    b1 = solve(dot(Ap, A) + lmda * I, dot(Ap, c0))
    if j == numreps - 1:
        ax.plot(index, b1, 'b-', lw=2, alpha=0.4, label='ridge estimates')
    else:
        ax.plot(index, b1, 'b-', lw=2, alpha=0.4)
    # Compute the standard least squares solution
    b2 = solve(dot(Ap, A), dot(Ap, c0))
    if j == numreps - 1:
        ax.plot(index, b2, 'g-', lw=2, alpha=0.5, label='OLS estimates')
    else:
        ax.plot(index, b2, 'g-', lw=2, alpha=0.5)


ax.plot(index, bstar, 'k-', lw=3, alpha=1, label='true value')
ax.legend(ncol=3, **legend_args)
ax.set_xlim(1, K)
plt.show()



