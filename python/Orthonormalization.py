#math and linear algebra stuff
import numpy as np

#plots
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (15.0, 15.0)
#mpl.rc('text', usetex = True)
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

"""
Gram-Schmidt method for basis orthonormalization
"""
size = 25
nbVec = 15
V = np.random.rand(size,nbVec)

#Naive orthogonalization loop
for i in np.arange(nbVec):
  #get current vector
  v=V[:,i]
  if i>0:
    #Compute projection onto other vector
    proj = np.dot(v,V[:,:i])
    #orthogonalize
    v = v - np.dot(V[:,:i],proj)
  #Normalize vector and update V
  V[:,i] = v / np.sqrt(np.dot(v,v))

#check that V is orthonormal
assert( np.allclose(np.identity(nbVec),np.dot(V.T,V)) )

"""
Eigen method for basis orthogonalization
"""
size = 25
nbVec = 15
V = np.random.rand(size,nbVec)

#Compute Gram matrix
Mg = np.dot(V.T,V)

#Perform eigendecomposition
D,Q=np.linalg.eig(Mg)

#Compute Mg2 with the modified vectors V2
V2 = np.dot(V,Q)
Mg2 = np.dot(V2.T,V2)

#check that Mg2 is orthonormal
assert( np.allclose(Mg2-Mg2*np.identity(nbVec),np.zeros((nbVec,nbVec))) )

#Show matrix Mg2
plt.imshow(Mg2,interpolation='none')

"""
Choleski method for basis orthonormalization
"""
size = 25
nbVec = 15
V = np.random.rand(size,nbVec)

#Compute Gram matrix
Mg = np.dot(V.T,V)

#Perform Choleski factorization
L=np.linalg.cholesky(Mg)

#Compute Mg2 with the modified vectors V2
V2 = np.dot(V,np.linalg.inv(L.T))
Mg2 = np.dot(V2.T,V2)

#check that Mg2 is orthonormal
assert( np.allclose(Mg2-Mg2*np.identity(nbVec),np.zeros((nbVec,nbVec))) )

#Show matrix Mg2
plt.imshow(Mg2,interpolation='none')



