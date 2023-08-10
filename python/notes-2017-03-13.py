import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from numpy.linalg import matrix_power as mpow
get_ipython().magic('matplotlib inline')

a = 2
b = 3
N = 100
x = np.random.rand(100)
noise = 0.1*np.random.randn(100)
y = a + b*x + noise

plt.scatter(x,y);

get_ipython().magic('pinfo np.hstack')

X = np.hstack((np.ones(N).reshape(N,1),x.reshape(N,1)))

X.shape

X[:5,:]

Y = y.reshape(N,1)

Y[:5,:]

A = la.solve(X.T @ X, X.T @ Y)

A

u = np.linspace(0,1,10)
v = A[0,0] + A[1,0]*u
plt.plot(u,v,'r',linewidth=4)
plt.scatter(x,y);

a = 3
b = 5
c = 8
N = 1000
x = 2*np.random.rand(N) - 1 # Random numbers in the interval (-1,1)
noise = np.random.randn(N)
y = a + b*x + c*x**2 + noise
plt.scatter(x,y,alpha=0.5,lw=0);

X = np.hstack((np.ones(N).reshape(N,1),x.reshape(N,1),(x**2).reshape(N,1)))

Y = y.reshape(N,1)

A = la.solve((X.T @ X),X.T @ Y)

u = np.linspace(-1,1,20)
v = A[0,0] + A[1,0]*u + A[2,0]*u**2
plt.plot(u,v,'r',linewidth=4)
plt.scatter(x,y);

