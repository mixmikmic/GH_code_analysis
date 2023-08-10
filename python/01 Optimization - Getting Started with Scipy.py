get_ipython().magic('pylab inline')
import numpy as np
from scipy import optimize

def convex(x):
    return -np.exp( -(x -3)**2)
def non_convex(x):
    return x**2 + 10*np.sin(x)

x = np.arange(-10, 10, 0.1)

f, plots = subplots(1, 2)
plots[0].plot(x, convex(x), label='Convex')
plots[0].grid()
plots[0].legend()
plots[1].plot(x, non_convex(x), label='Non-Convex')
plots[1].grid()
plots[1].legend()

