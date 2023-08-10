import numpy as np

def cholesky(A):
    A1 = A.copy()
    n = np.shape(A1)[0]
    R = np.zeros((n,n))
    for i in range(n):
        R[i,i] = np.sqrt(A1[i,i])
        u_t = (1/R[i,i])*A1[i,i+1:]
        U = np.outer(u_t,u_t)
        R[i,i+1:] = u_t 
        A1[i+1:,i+1:] = A1[i+1:,i+1:] - U
    return [R, R.T]

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

A = np.array([[4,-2,2],[-2,2,-4],[2,-4,11]])
print(A)

b = np.array([[3],[-7],[3]])
print(b)

[R,R_T] = cholesky(A)
print(R)
print(R_T)
print(A)

[c,x] = back_subs(R_T,R,b)
print(x)

A.dot(x)





