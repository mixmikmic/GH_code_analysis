# Import NumPy and seed random number generator to make generated matrices deterministic
import numpy as np
np.random.seed(1)

# Create a symmetric matrix with random entries
A = np.random.rand(4, 4)
A = A + A.T
print(A)

# Compute eigenvectors to generate a set of orthonormal vector
evalues, evectors = np.linalg.eig(A)

# Verify that eigenvectors R[i] are orthogonal (see Lecture 8 notebook)
import itertools
pairs = itertools.combinations_with_replacement(range(np.size(evectors, 0)), 2)
for p in pairs:
    e0, e1 = p[0], p[1]
    print("Dot product of eigenvectors vectors {}, {}: {}".format(e0, e1, evectors[:, e0].dot(evectors[:, e1])))

R = evectors.T

Ap = (R).dot(A.dot(R.T))
print(Ap)

print((R.T).dot(Ap.dot(R)))

