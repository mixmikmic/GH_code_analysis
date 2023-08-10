import numpy as np
import matplotlib.pyplot as plt

P = np.array([[0,0],[1,0],[2,0],[3,1]]).T
X = np.array([[2,1],[2,2],[2,3],[1,4]]).T
P = P + np.random.randn(*P.shape) * 0.1

def plotpoints(P,X):
    fig,ax = plt.subplots()
    ax.scatter(P[0,:],P[1,:], label = "$P$")
    ax.scatter(X[0,:],X[1,:], label = "$X$")
    ax.plot(np.vstack((P[0,:],X[0,:])),np.vstack((P[1,:],X[1,:])),
           color = "black", linewidth = 0.5, alpha = 0.2)
    ax.axis("equal")
    plt.legend();
    
plotpoints(P,X)

muP = np.mean(P, axis=1, keepdims=True)
muX = np.mean(X, axis=1, keepdims=True)
Pprime = P - muP
Xprime = X - muX
plotpoints(Pprime, Xprime)

W = np.zeros((2,2))
for i in range(P.shape[1]):
    W += Xprime[:,[i]] @ Pprime[:,[i]].T
W

# or equivalently (not obvious why, have a look at the docs of np.dot)
# W = np.dot(Xprime, Pprime.T)

U,S,V = np.linalg.svd(W)
R = U @ V.T
t = muX - (R @ muP)

print(R)
assert(np.allclose(np.linalg.inv(R),R.T)) # R^-1 == R^T
assert(np.isclose(np.dot(R[0,:], R[1,:]), 0)) # rows should be orthogonal
assert(np.isclose(np.linalg.det(R), 1)) # determinant should be +1

Paligned = R @ P + t
Paligned

# Same thing using homoegeneous coordinates
hmatrix = np.vstack((np.hstack((R,t)),[0,0,1]))
hP = np.vstack((P,np.ones(P.shape[1])))
hPaligned = hmatrix @ hP
Paligned = hPaligned[0:2,:]
Paligned

plotpoints(Paligned,X)

