def func(x):

    f = x[0] + x[1]
    c = x[0]**2 + x[1]**2 - 8
    
    return f, c

def quadpenalty(x, mu):
    
    f, c = func(x)
    P = mu/2.0*c**2
    return f + P

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import colormaps as cmaps

n = 100
x0 = np.linspace(-3.5, 3.5, n)
x1 = np.linspace(-3.5, 3.5, n)
[X0, X1] = np.meshgrid(x0, x1, indexing='ij')

F = np.zeros((n, n))
C = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        F[i, j], C[i, j] = func([X0[i, j], X1[i, j]])


plt.figure()
plt.contourf(X0, X1, F, 100, cmap=cmaps.viridis)
plt.colorbar()
plt.contour(X0, X1, C, levels=[0], linewidths=2, colors='k')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.show()

mu = 1.0

for i in range(n):
    for j in range(n):
        F[i, j] = quadpenalty([X0[i, j], X1[i, j]], mu)
        
plt.figure()
plt.contour(X0, X1, F, 100, cmap=cmaps.viridis)
plt.colorbar()
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.show()

mu = 100.0

for i in range(n):
    for j in range(n):
        F[i, j] = quadpenalty([X0[i, j], X1[i, j]], mu)
        
plt.figure()
plt.contour(X0, X1, F, 100, cmap=cmaps.viridis)
plt.colorbar()
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.show()

mu = 0.005

for i in range(n):
    for j in range(n):
        F[i, j] = quadpenalty([X0[i, j], X1[i, j]], mu)
        
plt.figure()
plt.contour(X0, X1, F, 100, cmap=cmaps.viridis)
plt.colorbar()
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.show()

import pyoptsparse

def opt(xdict):  # return both f and c

    x = xdict['x']  # uses a dictionary with whatever keys you define below
    f = quadpenalty(x, mu)
    
    outputs = {}
    outputs['obj'] = f 
    
    fail = False  # can use a flag to denote a failure, optimizer will try to recover and progress
    
    return outputs, fail


# starting point
x0 = [0.0, 0.0]

# define the problem.  Use same keys as above.
optProb = pyoptsparse.Optimization('penalty', opt)
optProb.addObj('obj')
optProb.addVarGroup('x', len(x0), type='c', value=x0)

# choose the solver, in this case SNOPT
opt = pyoptsparse.SNOPT()
opt.setOption('Major feasibility tolerance', 1e-6)
opt.setOption('Major optimality tolerance', 1e-6)
opt.setOption('iPrint', 6)  # normally you would not want to do this, but this notebook can't write files.  In general, you'll get two output files with detailed information.
opt.setOption('iSumm', 6)

# iterate
n = 20
muvec = np.logspace(-1, 3, n)
fstar = np.zeros(n)
xstar = np.zeros(n)

for i, mu in enumerate(muvec):
    sol = opt(optProb, sens='FD')  # finite difference

#     xstar[i] = sol.xStar['x'][0]  # just take one of x values since both are same
    fstar[i] = sol.fStar

with plt.style.context(('fivethirtyeight')):
    plt.figure()
    plt.semilogx(muvec, fstar)
    plt.xlabel('$\mu$')
    plt.ylabel('$f^*$')
    plt.ylim([-4.3, -3.95])

    plt.figure()
    plt.loglog(muvec, np.abs(fstar) - 4.0)
    plt.xlabel('$\mu$')
    plt.ylabel('$f^*$')

def augmented(x, mu, lam):
    
    f, c = func(x)
    P = mu/2.0*c**2
    return f + lam*c + P


mu = 0.005
lam = 0.2

n = 100
for i in range(n):
    for j in range(n):
        F[i, j] = quadpenalty([X0[i, j], X1[i, j]], mu)
        
plt.figure()
plt.contour(X0, X1, F, 100, cmap=cmaps.viridis)
plt.colorbar()
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.show()


n = 100
for i in range(n):
    for j in range(n):
        F[i, j] = augmented([X0[i, j], X1[i, j]], mu, lam)
        
plt.figure()
plt.contour(X0, X1, F, 100, cmap=cmaps.viridis)
plt.colorbar()
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.show()

