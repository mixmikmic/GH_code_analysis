import numpy as np

def sweep(A, k):
    """
    Perform a SWEEP operation on A with the pivot element A[k,k].
    
    :param A: a square matrix.
    :param k: the pivot element is A[k, k].
    :returns a swept matrix. Original matrix is unchanged.
    """
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError('A is not a square array.')
    if k >= n or k < 0:
        raise IndexError('k is not a valid row index for pivot element.')
        
    #  Fill with the general formula
    B = A - np.outer(A[:, k], A[k, :]) / A[k, k]
    
    # Modify the k-th row and column
    B[k, :] = A[k, :] / A[k, k]
    B[:, k] = A[:, k] / A[k, k]
    
    # Modify the pivot
    B[k, k] = -1 / A[k, k]
    return B

# Use our sweep function.
A = np.array([[1,2,3],[7,11,13],[17,21,23]], dtype=float)

# Perform sweep operator repeatedly.
det_A = 1
A_swp = A.copy()
for k in (0, 1, 2):
    det_A *= A_swp[k, k]
    A_swp = sweep(A_swp, k)
    
print 'A_swp:\n', A_swp
print 'Determinant of A:', det_A

#======================================
# Generate a testing data.
#======================================
n, p = 100, 5

# Generate a random n-by-p data matrix.
X = np.random.normal(0, 1, (n, p))

# Assume the real coefficients are 1 ... p. Intercept is 0.
# This is the ground-truth we want to evaluate our code against.
beta = np.array(range(1, p+1))

# Synthesis the output Y.
Y = np.dot(X, beta)

#======================================
# Solve by scipy.
#======================================
from scipy import linalg
coef, resid, rank, sigma = linalg.lstsq(X, Y)
print '----- SCIPY -----'
print '[scipy] coefficients:', coef.round(6)
print '[scipy]      residue:', resid.round(6)

#======================================
# Solve by sweep.
#======================================
# Stack an additional n-by-1 ones vector for solving intercepts.
Z = np.hstack((np.ones(n).reshape((n, 1)), X, Y.reshape((n, 1))))
A = np.dot(Z.T, Z)

S = A
for k in range(p+1): # +1 because we added one intercept column.
    S = sweep(S, k)
    
beta = S[:p+1, -1]
rss = S[-1, -1]
print '----- SWEEP -----'
print '[sweep] coefficients:', beta.round(6)
print '[sweep]     residure:', rss.round(6)



