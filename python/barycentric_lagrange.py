get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
import numpy as np
from matplotlib import pyplot as plt

def fun(x):
    f = 1.0/(1.0+16.0*x**2)
    return f

def LagrangeInterpolation(X,Y,x):
    nx = np.size(x)
    nX = np.size(X)
    # compute the weights
    w = np.ones(nX)
    for i in range(nX):
        for j in range(nX):
            if i != j:
                w[i] = w[i]/(X[i]-X[j])
    # Evaluate the polynomial at x
    num= np.zeros(nx)
    den= np.zeros(nx)
    eps=1.0e-14
    for i in range(nX):
        num = num + Y[i]*w[i]/((x-X[i])+eps)
        den = den + w[i]/((x-X[i])+eps)
    f = num/den
    return f

xmin, xmax = -1.0, +1.0
N = 15 # Degree of polynomial

X = np.linspace(xmin,xmax,N+1)
Y = fun(X)
x = np.linspace(xmin,xmax,100)
fi = LagrangeInterpolation(X,Y,x)
fe = fun(x)
plt.plot(x,fe,'b--',x,fi,'r-',X,Y,'o')
plt.title('Degree '+str(N)+' using uniform points')
plt.legend(("True function","Interpolation","Data"),loc='lower center');

X = np.cos(np.linspace(0.0,np.pi,N+1))
Y = fun(X)
x = np.linspace(xmin,xmax,100)
fi = LagrangeInterpolation(X,Y,x)
fe = fun(x)
plt.plot(x,fe,'b--',x,fi,'r-',X,Y,'o')
plt.title('Degree '+str(N)+' using Chebyshev points')
plt.legend(("True function","Interpolation","Data"),loc='upper right');

