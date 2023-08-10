import numpy as np

def qr(A):
    n, m = A.shape
    R = A.copy()
    Q = np.eye(n)

    for k in range(m-1):
        x = np.zeros((n, 1))
        x[k:, 0] = R[k:, k]
        s = -1 * np.sign(x[k, 0])
        v = x
        v[k] = x[k] - s*np.linalg.norm(x)
        u = v / np.linalg.norm(v)
        
        R -= 2 * np.dot(u, np.dot(u.T, R))
        Q -= 2 * np.dot(u, np.dot(u.T, Q))
    return Q.T, R

# Test our QR function
A = np.array([[-2.0, 2, 3],
              [1, 3, 5],
              [-3, -1, 2]])

Q, R = qr(A)

print '[myqr] Q'
print Q.round(8)
print '[myqr] R'
print R.round(8)

Q_gt, R_gt = np.linalg.qr(A)
print '[numpy] Q'
print Q_gt
print '[numpy] R'
print R_gt

# Synthesis a dataset with n observations and p predictors.
n = 100
p = 5
X = np.random.random_sample((n, p))

# True coefficients are 1, 2, ..., p.
beta = np.array(range(1, p+1))
Y = np.dot(X, beta) + np.random.standard_normal(n)

# Stack (X Y) ans solve it by QR decomposition.
# Here we add the first column to be 1's for solving the intercepts.
Z = np.hstack((np.ones(n).reshape((n, 1)), X, Y.reshape((n, 1))))
_, R = qr(Z)
R1 = R[:p+1, :p+1]
Y1 = R[:p+1, p+1]

# Solve beta.
beta = np.linalg.solve(R1, Y1)
print beta



