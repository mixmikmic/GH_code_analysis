# Import NumPy and seed random number generator to make generated matrices deterministic
import numpy as np
np.random.seed(1)

# Create a matrix with random entries
A = np.random.rand(4, 4)

# Use QR factorisation of A to create an orthogonal matrix Q (QR is covered in IB)
Q, R = np.linalg.qr(A, mode='complete')

print(Q.dot(Q.T))

import itertools

# Build pairs (0,0), (0,1), . . . (0, n-1), (1, 2), (1, 3), . . . 
pairs = itertools.combinations_with_replacement(range(len(Q)), 2)

# Compute dot product of column vectors q_{i} \cdot q_{j}
for p in pairs:
    col0, col1 = p[0], p[1]
    print ("Dot product of column vectors {}, {}: {}".format(col0, col1, Q[:, col0].dot(Q[:, col1])))

# Compute dot product of row vectors q_{i} \cdot q_{j}
pairs = itertools.combinations_with_replacement(range(len(Q)), 2)
for p in pairs:
    row0, row1 = p[0], p[1]
    print ("Dot product of row vectors {}, {}: {}".format(row0, row1, Q[row0, :].dot(Q[row1, :])))

print("Determinant of Q: {}".format(np.linalg.det(Q)))

