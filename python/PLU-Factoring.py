import numpy as np

def LU(A,b):
    n = b.size
    L = np.zeros((n,n))
    U = np.zeros((n,n))
    for j in range(0,n-1):
        for i in range(j+1,n):
            mult = A[i,j]/A[j,j]
            for k in range(j,n):
                A[i,k] = A[i,k] - mult*A[j,k]
                L[k,k] = 1
                U[k,k] = A[k,k]
            b[i] = b[i] - mult*b[j]
            L[i,j] = mult
            U[j,i] = A[j,i]
    return [L,U]

def back_subs(L,U,b):
    n = b.size
    b_temp = b
    x = np.ones(n)
    c = np.ones(n)
    
    for l in range(0,n):
        for m in range(l+1,n):
            b_temp[l] = b_temp[l] - L[l,m]*c[m]
        c[l] = b[l]/L[l,l]
        
    for l in range(n-1,-1,-1):
        for m in range(l+1,n):
            b_temp[l] = b_temp[l] - U[l,m]*x[m]
        x[l] = b[l]/A[l,l]
    return [c,x]

A = np.array([[3, 1, 2],[6, 3, 4], [3, 1 , 5]])
print(A)
b = np.array([[0,1,3]]).T
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









