import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

M = np.random.randint(0,10,size=(4,3))

print(M)

np.sum(M)

M.sum()

np.sum(M,axis=0) # Sum of the columns

np.sum(M,axis=1) # Sum of the rows

M.sum(axis=1)

print(M)

M.mean(axis=0)

M.mean(axis=1)

M.prod(axis=0)

N = np.random.randint(-100,100,size=(3,7))
print(N)

N[2,4]

N[0,0]

c = N[:,3]

print(c)

c.ndim

c.shape

type(c)

C = c.reshape(3,1)

print(C)

C.ndim

C.shape

P = np.random.randint(-50,50,size=(4,4))

print(P)

D = P[2:4,2:4]
print(D)

A = np.ones((2,2))
B = 2*np.ones((2,2))
C = 3*np.ones((2,2))
D = 4*np.ones((2,2))

print(A)

print(D)

X1 = np.vstack([A,C])
X2 = np.vstack([B,D])
X = np.hstack([X1,X2])
print(X)

v1 = np.random.randint(0,10,10)
print(v1)
v2 = np.random.randint(0,10,10)
print(v2)

B = np.vstack([v1,v2])
print(B)

C = np.vstack([v1,v2]).T
print(C)

