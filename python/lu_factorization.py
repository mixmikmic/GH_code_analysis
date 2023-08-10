import numpy as np

def LU(A,b):
    n = b.size
    L = np.zeros((n,n))
    U = np.zeros((n,n))
    for k in range(0,n):
        for r in range(0,n):
            U[k,r] = A[k,r] 
            if k==r:
                L[k,r] = 1
            if k<r:
                factor = A[r,k]/A[k,k]
                L[r,k] = factor
                for c in range(0,n):
                    A[r,c] = A[r,c] - factor*A[k,c]
                    U[r,c] = A[r,c]
    return [L,U]

def back_subs(L,U,b):
    n = b.size
    x = np.zeros(n)
    c = np.zeros(n)
    
    c[0] = b[0]/L[0,0]
    for l in range(1,n):
        s = 0
        for m in range(0,l):
            s = s + L[l,m]*c[m]
        c[l] = (b[l] - s)/L[l,l]
    
    for l in range(n-1,-1,-1):
        t = 0
        for m in range(l+1,n):
            t = t + U[l,m]*x[m]
        x[l] = (c[l] - t)/U[l,l]
    return [c,x]

A = np.array([[1,2,-1],[2,1,-2],[-3,1,1]])
print(A)
b = np.array([[3],[3],[-6]])
print(b)

[L,U] = LU(A,b)
print("L = ")
print(L)
print("U = ")
print(U)

L.dot(U)

[c,x] = back_subs(L,U,b)
print("x = ")
print(x)
print("c = ")
print(c)

(L.dot(U)).dot(x)



