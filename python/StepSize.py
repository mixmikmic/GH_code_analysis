get_ipython().magic('matplotlib inline')
import numpy as np
from math import sin, cos, exp
import matplotlib.pyplot as plt

# just a simple 1D function to illustarate
def f(x):
    return exp(x)*sin(x)

# these are the exact derivatives so we can compare performance
def g(x):
    return exp(x)*sin(x) + exp(x)*cos(x)

# let's take a bunch of different step sizes from very large to very small
n = 26
step_size = np.logspace(0, -25, n)

# initialize results array (forward difference, central difference)
grad_fd = np.zeros(n)
grad_cd = np.zeros(n)

# arbitrarily chosen point
x = 0.5

# loop through and try all the different starting points
for i in range(n):
    h = step_size[i]
    grad_fd[i] = (f(x + h) - f(x))/h  
    grad_cd[i] = (f(x + h) - f(x-h))/(2*h)

# compute relative error compared to the exact solution
grad_exact = g(x)
error_fd = np.abs((grad_fd - grad_exact)/grad_exact)
error_cd = np.abs((grad_cd - grad_exact)/grad_exact)

plt.style.use('ggplot')
plt.figure()
plt.loglog(step_size, error_fd, '.-', label='forward')
plt.loglog(step_size, error_cd, '.-', label='central')
plt.gca().set_ylim(ymin=1e-18, ymax=1e1)
ticks = np.arange(-1, -26, -3)
plt.xticks(10.0**ticks)
plt.gca().invert_xaxis()
plt.legend(loc='center right')
plt.xlabel('step size')
plt.ylabel('relative error')
plt.show()

from cmath import sin, cos, exp

# initialize
grad_cs = np.zeros(n)

# loop through each step size
for i in range(n):
    h = step_size[i]
    grad_cs[i] = f(x + complex(0, h)).imag / h  
    

# compute error
error_cs = np.abs((grad_cs - grad_exact)/grad_exact)

# the error is below machine precision in some cases so just add epsilon error so it shows on plot
error_cs[error_cs == 0] = 1e-16

plt.figure()
plt.loglog(step_size, error_fd, '.-', label='forward')
plt.loglog(step_size, error_cd, '.-', label='central')
plt.loglog(step_size, error_cs, '.-', label='complex')
plt.gca().set_ylim(ymin=1e-18, ymax=1e1)
ticks = np.arange(-1, -26, -3)
plt.xticks(10.0**ticks)
plt.gca().invert_xaxis()
plt.legend(loc='center right')
plt.xlabel('step size')
plt.ylabel('relative error')
plt.show()

