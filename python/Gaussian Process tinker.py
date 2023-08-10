from autograd import grad
import autograd.numpy as np
import autograd.numpy.random as npr
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

xmin,xmax=0,10

n_basis_fxns=15
basis_points=np.linspace(xmin,xmax,n_basis_fxns)
basis_fxns = np.empty(n_basis_fxns,dtype=object)

class RBF():
    def __init__(self,center,r=1.0):
        self.c=center
        self.r=r

    def __call__(self,x):
        return np.exp(-(np.sum((x-self.c)**2) / (2*self.r**2)))

class StepFxn():
    def __init__(self,center,r=1.0):
        self.c=center
        self.r=r
    
    def __call__(self,x):
        return 1.0*(np.abs(x-self.c) < self.r)

#print(basis_points)
basis_expansion_of_1 = np.zeros(n_basis_fxns)

for i,c in enumerate(basis_points):
    
    basis_fxns[i]=RBF(c,0.5)
    #basis_fxns[i] = StepFxn(c)
    #print(phi(1.0))
    basis_expansion_of_1[i] = basis_fxns[i](1.0)

plt.plot(basis_expansion_of_1)

for fxn in basis_fxns:
    print(fxn(1.0))

basis_fxns

def basis_fxn_expansion(x):
    return np.array([f(x) for f in basis_fxns])

plt.plot(basis_fxn_expansion(1.0))

def pred_y(x,weights):
    phis = basis_fxn_expansion(x)
    return np.dot(weights,phis)

x = np.linspace(xmin,xmax,500)
def f(x):
    return np.sin(2*x) - np.sin(3*x)
y =  f(x)

plt.plot(x,y)

pred_y(5,np.ones(len(basis_fxns)))

Q = np.array([basis_fxn_expansion(i) for i in x])
plt.imshow(Q)

np.dot(Q.T,np.ones(len(x))).shape

def objective(weights):
    ''' minimize this'''
    #prior = np.sum(weights**2)
    #pred = np.array([pred_y(p,weights) for p in x])
    pred = np.dot(Q,weights)
    mse = np.sum((pred-y)**2)
    return mse
    #log_likelihood = - np.sum((pred-y)**2)
    #return prior + log_likelihood

from time import time
t = time()
objective(np.ones(len(basis_fxns)))
print(time() - t)

grad(objective)(np.ones(len(basis_fxns)))

def gradient_descent(objective,start_point,
                     n_iter=100,step_size=0.1):
    intermediates = np.zeros((n_iter,len(start_point)))
    gradient = grad(objective)
    x = start_point
    intermediates[0] = x
    for i in range(1,n_iter):
        update = gradient(x)*step_size
        x = x - update
        intermediates[i] = x
    return intermediates

w = gradient_descent(objective,npr.randn(len(basis_fxns)),
                        n_iter=10000,step_size=0.005)
w_opt = w[-1]

plt.plot(w_opt)

predicted = np.array([pred_y(i,w_opt) for i in x])
plt.plot(x,predicted,label='Model',linewidth=3,alpha=0.5)
plt.plot(x,y,label='True',linewidth=3,alpha=0.5)
for i,basis_fxn in enumerate(basis_fxns):
    if i == 0:
        plt.plot(x,[basis_fxn(x_) for x_ in x],c='grey',alpha=0.3,
                label='Basis functions')
        plt.plot(x,[basis_fxn(x_)*w_opt[i] for x_ in x],c='grey',
                 label='Weighted basis functions')
    else:
        plt.plot(x,[basis_fxn(x_)*w_opt[i] for x_ in x],c='grey')
        plt.plot(x,[basis_fxn(x_) for x_ in x],c='grey',alpha=0.3)

#plt.ylim(-3,2)
plt.legend(loc='best')

mse = np.sum((predicted - y)**2) / len(predicted)
mse

import gptools







# reproducing Figure 1 from: http://mlg.eng.cam.ac.uk/pub/pdf/Ras04.pdf
import numpy as np
import numpy.random as npr
from numpy.linalg import cholesky
from numpy.matlib import repmat

xs = np.linspace(-5,5,1000)
ns = len(xs)
keps=1e-9

m = lambda x: 0.25*x**2
def K_mat(xs_1,xs_2):
    diff_mat = repmat(xs_1,len(xs_2),1) - repmat(xs_2,len(xs_1),1).T
    return np.exp(-0.5*(diff_mat)**2)

fs = m(xs) + cholesky(K_mat(xs,xs)+keps*np.eye(ns)).dot(npr.randn(ns))

npr.seed(0)
mean = m(xs)
choleskied = cholesky(K_mat(xs,xs)+keps*np.eye(ns))



fs = np.zeros((10000,len(xs)))

for i in range(len(fs)):
    fs[i] = mean + choleskied.dot(npr.randn(ns))
    if i < 5:
        plt.plot(xs,fs[i],c='blue',alpha=0.5)

plt.plot(xs,fs.mean(0),c='cyan')
plt.fill_between(xs,fs.mean(0)+1.96*np.sqrt(fs.std(0)),fs.mean(0)-1.96*np.sqrt(fs.std(0)),alpha=0.2)

plt.hist([xs[np.argmin(f)] for f in fs],bins=50);



get_ipython().magic('timeit np.argmin(cholesky.dot(npr.randn(ns)))')

np.argmin(cholesky.dot(npr.randn(ns,100)))

cholesky.dot(npr.randn(ns,100)).shape

la.cholesky(K_mat(xs,xs)+keps*np.eye(ns)).shape

plt.imshow(K_mat(xs,xs),cmap='Blues',interpolation='none');
plt.colorbar()
plt.figure()

plt.imshow(la.cholesky(K_mat(xs,xs)+keps*np.eye(ns)),cmap='Blues',interpolation='none');
plt.colorbar()

# what's the minimum in this draw?
xs[np.argmin(fs)]



