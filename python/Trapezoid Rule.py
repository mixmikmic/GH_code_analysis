get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return (x-3)*(x-5)*(x-7)+85

x = np.linspace(0, 10, 200)
y = f(x)

a, b = 1, 8 # the left and right boundaries
N = 5 # the number of points
xint = np.linspace(a, b, N)
yint = f(xint)

plt.plot(x, y, lw=2)
plt.axis([0, 9, 0, 140])
plt.fill_between(xint, 0, yint, facecolor='gray', alpha=0.4)
plt.text(0.5 * (a + b), 30,r"$\int_a^b f(x)dx$", horizontalalignment='center', fontsize=20);

from __future__ import print_function
from scipy.integrate import quad
integral, error = quad(f, a, b)
integral_trapezoid = sum( (xint[1:] - xint[:-1]) * (yint[1:] + yint[:-1]) ) / 2
print("The integral is:", integral, "+/-", error)
print("The trapezoid approximation with", len(xint), "points is:", integral_trapezoid)

