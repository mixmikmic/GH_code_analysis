import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

A = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(A)

v = A[:,1]
print(v)

A.ndim

A.shape

v.ndim

v.shape

v = v.reshape(3,1)

v

v.ndim

v.shape

A = np.array([[1,-3],[-3,5]])
evals,evecs = la.eig(A)

print(evals)

evals.dtype

evals = evals.real

print(evals)

print(evecs)

t = np.linspace(0,5,50)
y0 = evecs[0,0]*np.exp(evals[0]*t)
y1 = evecs[1,0]*np.exp(evals[0]*t)
plt.plot(t,y0,t,y1)
plt.legend(('$y_0(t)$','$y_1(t)$'))
plt.show()

b = np.array([[1],[1]])
C = la.solve(evecs,b)

print(C)

