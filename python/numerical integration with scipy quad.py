import numpy as np
from scipy import integrate

def f(x):
    """simple function to integrate"""
    return np.sin(x)


def trap(f, xmin, xmax, npoints=10):
    """
    computes the integral of f using trapezoid rule
    """
    area = 0
    x = np.linspace(xmin, xmax, npoints)
    N = len(x)
    dx = x[1] - x[0]
    
    for k in range(1, N):
        area += (f(x[k - 1]) + f(x[k])) * dx / 2
        
    return area

get_ipython().magic('timeit trap(f, 0, np.pi, 1500000) - 2   # 2 is the actual integral value')

get_ipython().magic('timeit integrate.quad(f, 0, np.pi)')



