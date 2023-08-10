import numpy as np

def eigen_qr(A):
    """An implementation of eigen decomposition using QR decompostion.
    
    Args:
        A: A square matrix.
        
    Returns:
        D: Eigen values. 
        V: A matrix whose columns are eigen vectors.
    """
    T = 1000
    r, c = A.shape

    V = np.random.random_sample((r, r))

    for i in range(T):
        Q, _ = np.linalg.qr(V)   # orthogonalize V
        V = np.dot(A, Q)         # update V

    Q, R = np.linalg.qr(V)

    return R.diagonal(), Q

# fix the random seed.
np.random.seed(1)

n = 100
p = 5
X = np.random.random_sample((n, p))
A = np.dot(X.T, X)

# Use the eigen_qr implemented above.
D, V = eigen_qr(A)
print 'eigen values  [ours]:', D.round(6)
print 'eigen vectors [ours]:\n',V.round(6)

# Compare the result with the numpy calculation.
print '=' * 10
eigen_value_gt, eigen_vector_gt = np.linalg.eig(A)
print 'eigen values  [numpy]:', eigen_value_gt.round(6)
print 'eigen vectors [numpy]:\n', eigen_vector_gt.round(6)



