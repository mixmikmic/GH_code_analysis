# Import NumPy and seed random number generator to make generated matrices deterministic
import numpy as np
np.random.seed(1)

# Create a symmetric matrix with random entries
A = np.random.rand(5, 5)
A = A + A.T
print(A)

# Compute eigenvectors of A
evalues, evectors = np.linalg.eig(A)

print("Eigenvalues: {}".format(evalues))
print("Eigenvectors: {}".format(evectors))

import itertools

# Build pairs (0,0), (0,1), . . . (0, n-1), (1, 2), (1, 3), . . . 
pairs = itertools.combinations_with_replacement(range(len(evectors)), 2)

# Compute dot product of eigenvectors x_{i} \cdot x_{j}
for p in pairs:
    e0, e1 = p[0], p[1]
    print ("Dot product of eigenvectors {}, {}: {}".format(e0, e1, evectors[:, e0].dot(evectors[:, e1])))

print("Testing  Ax and (lambda)x: \n {}, \n {}".format(A.dot(evectors[:,1]), evalues[1]*evectors[:,1]))

B = np.random.rand(5, 5)
evalues, evectors = np.linalg.eig(B)

print("Eigenvalues: {}".format(evalues))
print("Eigenvectors: {}".format(evectors))

# Compute dot product of eigenvectors x_{i} \cdot x_{j}
pairs = itertools.combinations_with_replacement(range(len(evectors)), 2)
for p in pairs:
    e0, e1 = p[0], p[1]
    print ("Dot product of eigenvectors {}, {}: {}".format(e0, e1, evectors[:, e0].dot(evectors[:, e1])))

