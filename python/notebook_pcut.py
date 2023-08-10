# Load libraries

# Math
import numpy as np

# Import data
import scipy.io

# Check random generator
randnb = np.random.uniform(0,2**53)
print(randnb)

# Load 20NEWS dataset
mat = scipy.io.loadmat('datasets/20NEWS.mat')
A = mat['A'] # scipy.sparse.csc.csc_matrix
n = W.shape[0]
Cgt = mat['C'] - 1; Cgt = Cgt.squeeze()
nc = len(np.unique(Cgt))
print(n,nc)

# Global parameters
nc = 20
speed = 5
alpha = 0.95
maxiter = 500

# Symmetrize W
W = A
bigger = W.T > W
W = W - W.multiply(bigger) + W.T.multiply(bigger)
#print((W-W.transpose()).sum())

# Degree vector, matrix
D = W.sum(axis=0)
#print(D.shape,D[:10])
deg = scipy.sparse.spdiags(D,0,n,n)
#print(type(deg))

# Inverse degree matrix
Dinv = 1/D
Dinv = Dinv.squeeze()
#print(Dinv.shape)
idg = scipy.sparse.spdiags(Dinv,0,n,n)

# Random Walk Matrix
RW = idg*W
W = RW
#print(W[:100,:5])

# Coarsening parameters
collapse = 0
numlevel = 20

# cell array for system matrices at each level
Ws = RW
Rs = [] # restrictions
Ps = [] # prolongations
I = scipy.sparse.identity(n, dtype=W.dtype)
Sys = I - alpha* RW  # the (I - alpha D^(-1)W ) version
#print(type(Sys)) # scipy.sparse.csr.csr_matrix

# Coarsening
global ll; global uu; 
ll = []; uu = []

uu = scipy.sparse.triu(Sys,1,format="csr")  # store the upper and lower triangular portions 
ll = Sys - uu                               # of the system matrix at each level.
#print(uu[:5,:5])
#print(ll[:5,:5])







