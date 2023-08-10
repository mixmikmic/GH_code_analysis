# this cell is skipped.
# it is to show the polynomial and do some basic plotting setup

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

get_ipython().magic("config InlineBackend.figure_format = 'svg'")
plt.rcParams['figure.figsize'] = (14, 5.5)
plt.rcParams.update({'font.size': 15})

def fun(x, a=1, b=-7, c=5, d=13):
    return a * x ** 3 + b * x ** 2 + c * x + d

def make_plot():
    x = np.linspace(-2, 7, 1000)
    plt.plot(x, fun(x), label=r'$f(x)$', lw=3)
    plt.axhline(0, color='gray', lw=0.5)
    plt.yticks([-30, 0 , 30])
    plt.legend()
    plt.ylim(-30, 30)
    plt.show()

make_plot()

import numpy as np
from scipy import optimize

def fun(x, a=1, b=-7, c=5, d=13):
    return a * x ** 3 + b * x ** 2 + c * x + d

x = np.linspace(-2, 7, 1000)
sol = optimize.root(fun, [-2, 2, 6])

plt.plot(x, fun(x), lw=3)
plt.plot(sol.x, fun(sol.x), 'd', ms=10)
plt.axhline(0, color='gray', lw=0.5)
plt.show()

sol = optimize.root(fun, [-2, 10, 6])
sol.x

