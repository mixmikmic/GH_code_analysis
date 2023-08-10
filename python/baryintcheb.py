get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
import numpy as np
from matplotlib import pyplot as plt

def fun(x):
    f = 1.0/(1.0+16.0*x**2)
    return f

def BaryInterp(X,Y,x):
    nx = np.size(x)
    nX = np.size(X)
    f  = 0*x
    # Compute weights
    w  = (-1.0)**np.arange(0,nX)
    w[0]    = 0.5*w[0]
    w[nX-1] = 0.5*w[nX-1]
    # Evaluate barycentric foruma at x values
    for i in range(nx):
        num, den = 0.0, 0.0
        for j in range(nX):
            if np.abs(x[i]-X[j]) < 1.0e-15:
                num = Y[j]
                den = 1.0
                break
            else:
                num += Y[j]*w[j]/((x[i]-X[j]))
                den += w[j]/(x[i]-X[j])
        f[i] = num/den
    return f

xmin, xmax = -1.0, +1.0
N = 19 # degree of polynomial

X = np.cos(np.linspace(0.0,np.pi,N+1))
Y = fun(X)
x = np.linspace(xmin,xmax,100)
fi = BaryInterp(X,Y,x)
fe = fun(x)
plt.figure(figsize=(8,5))
plt.plot(x,fe,'b--',x,fi,'r-',X,Y,'o')
plt.legend(("True function","Interpolation","Data"),loc='upper right')
plt.axis([-1.0,+1.0,0.0,1.1]);

