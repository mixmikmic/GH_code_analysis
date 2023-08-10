import numpy as np
import scipy as sp
import scipy.linalg
import time
import matplotlib.pyplot as plt

#setting the size of the arrays for testing, will start small for testing and then will scale up for timing
N = 1000

#sigmas exist on a uniform distribution from [100, 300]

#pure diagonal matrix with equal values
def makeV1(N, sigmarange=[100,300]):
    sigma1 = np.random.random()*(sigmarange[1]-sigmarange[0]) + sigmarange[0]
    V1 = sigma1 * np.identity(N)
    return V1
V1 = makeV1(N)
#checks to ensure symmetry (and hermicity since all real)
print np.allclose(V1,np.ndarray.transpose(V1))

#diagonal matrix with different elements
def makeV2(N, sigmarange=[100,300]):
    sigma2 = np.random.rand(N)*(sigmarange[1]-sigmarange[0]) + sigmarange[0]
    V2 = sigma2 * np.identity(N)
    return V2
V2 = makeV2(N)
print np.allclose(V2,np.ndarray.transpose(V2))



#computes matrix square root of inverse variance matrix provided variance matrix
def matrixsqrt(V, label="0"):
    start = time.time()
    N = len(V[0])
    wt = np.empty([N,N]) #square root matrix (transposed)
    logdet = 0.
    #extracts eigenvalues and eigenvectors (bottleneck!)
    eigs, eigvecs = sp.linalg.eigh(V)
    for i in range(N):
        #sets each column in our transposed square root to the eigenvalue scaled by 1/sqrt(eig)
        wt[:,i] = (1./np.sqrt(eigs[i])) * eigvecs[:,i]
        logdet += np.log(2 * np.pi * eigs[i])
        #transposes the result
    w = np.ndarray.transpose(wt)
    end = time.time()
    dt = end-start
    if(label!="0"):
        print("Time elapsed for " + label + " is: " + str(dt) + "s")
    return w, logdet, dt

def ismsqrt(w,V):
    N = len(V[0])
    if(np.allclose(np.dot(V, np.dot(np.ndarray.transpose(w), w)),np.identity(N))):
        return True
    else:
        return False

w1, logdet1, dt1 = matrixsqrt(V1, label="V1")
print ismsqrt(w1,V1)


w2, logdet2, dt2 = matrixsqrt(V2, label="V2")
print ismsqrt(w2,V2)

def makeV3(N, sigmarange=[100,300], A=300, tau=30):
    K = np.empty([N,N])
    t = np.arange(N, dtype='int')
    for i in range(N):
        for j in range(N):
            K[i][j] = A * np.exp(-0.5 * (t[i]-t[j])**2 / (tau **2))
    V3 = makeV2(N, sigmarange=sigmarange) + K
    return V3
V3 = makeV3(N)
print np.allclose(V3,np.ndarray.transpose(V3))

w3, logdet3, dt3 = matrixsqrt(V3, label="V3")
#Didn't print the matrix because it's too big and contains small off-diagonal elements 
print ismsqrt(w3,V3)

import time

NArray = np.logspace(1, 14, num=14, base=2, dtype='int')
#dt1 = np.empty(len(NArray))
#dt2 = np.empty(len(NArray))
dt3 = np.empty(len(NArray))
w = []
V = []
print "Generating Arrays..."
for i in range(len(NArray)):
    print i
    w.append(np.empty([NArray[i],NArray[i]]))
    start = time.time()
    V.append((makeV3(NArray[i])))
    dt = time.time() - start
    if (dt>2000):
        break;
print "Done"


print "Computer Sqare Root..."
for i in range(len(V)):
#    w1, logdet1, dt1[i] = matrixsqrt(makeV1(NArray[i]))
#    w2, logdet2, dt2[i] = matrixsqrt(makeV2(NArray[i]))
    w[i], logdet3, dt3[i] = matrixsqrt(V[i])
    print(str(i) + '/' + str(len(NArray)-1) + ",\t Size:" + str(NArray[i]) + ",\t Time:" + str(dt3[i]))
    if(dt3[i]>(2**10)):
        break;
print "DONE"

fig, ax = plt.subplots()
ax.set_xscale('log', basex=2)
ax.set_yscale('log', basey=2)
ax.set_ylabel("Excecution time (s)")
ax.plot(NArray, dt3, 'g-', label='t(actual)/D')
ax.plot(NArray, ((2**-8)*NArray)**2, 'b-', label='D^2')
ax.plot(NArray, ((2**-6)*NArray)**3, 'r-', label='D^3')
ax.legend()
plt.show()

print w1.shape

