import numpy as np

def LU(A):
    n = A.shape[0]
    L = np.identity(n)
    P = np.arange(n,dtype=int) # Permutation matrix
    U = np.array(A)
    for k in range(n-1):
        i = np.argmax(np.abs(U[k:n,k])) + k
        U[[k,i],k:n] = U[[i,k],k:n] # swap row i and k
        L[[k,i],0:k] = L[[i,k],0:k] # swap row i and k
        P[[k,i]] = P[[i,k]]         # swap row i and k
        for j in range(k+1,n):
            L[j,k] = U[j,k]/U[k,k]
            U[j,k:n] = U[j,k:n] - L[j,k]*U[k,k:n]
    return L,U,P

def LUSolve(L,U,P,b):
    n = L.shape[0]
    # solve Ly = Pb
    pb = b[P]
    y = np.empty_like(b)
    for i in range(n):
        y[i] = (pb[i] - L[i,0:i].dot(y[0:i]))/L[i,i]
    #solve Ux = y
    x = np.empty_like(b)
    for i in range(n-1,-1,-1):
        x[i] = (y[i] - U[i,i+1:n].dot(x[i+1:n]))/U[i,i]
    return x

n = 3
A = np.random.rand(n,n)
L,U,P = LU(A)
print "A = "; print A
print "L = "; print L
print "U = "; print U
print "P = "; print P

# Create a permutation matrix from the P vector
Pm = np.zeros((n,n))
for i in range(n):
    Pm[i,P[i]] = 1.0
print "Pm= "; print Pm
print "PA-LU = "; print Pm.dot(A) - L.dot(U)

b = np.random.rand(n)
x = LUSolve(L,U,P,b)
print A.dot(x) - b

