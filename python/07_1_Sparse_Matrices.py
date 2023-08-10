import numpy as np

# Create a random array with a lot of zeros
X = np.random.random((10, 5))
print(X)

X[X < 0.7] = 0
print(X)

from scipy import sparse

# turn X into a csr (Compressed-Sparse-Row) matrix
X_csr = sparse.csr_matrix(X)
print(X_csr)

# convert the sparse matrix to a dense array
print(X_csr.toarray())

# Sparse matrices support linear algebra:
y = np.random.random(X_csr.shape[1])
z1 = X_csr.dot(y)
z2 = X.dot(y)
np.allclose(z1, z2)

# Create an empty LIL matrix and add some items
X_lil = sparse.lil_matrix((5, 5))

for i, j in np.random.randint(0, 5, (15, 2)):
    X_lil[i, j] = i + j

print(X_lil)
print(X_lil.toarray())

X_csr = X_lil.tocsr()
print(X_csr)

from scipy.sparse import bsr_matrix

indptr = np.array([0, 2, 3, 6])
indices = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6]).repeat(4).reshape(6, 2, 2)
bsr_matrix((data,indices,indptr), shape=(6, 6)).toarray()

from scipy.sparse import dok_matrix
S = dok_matrix((5, 5), dtype=np.float32)
for i in range(5):
    for j in range(i, 5):
        S[i,j] = i+j
        
S.toarray()

