get_ipython().magic('matplotlib notebook')
from __future__ import division
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

sym.init_printing()

x1, x2, x = sym.symbols("x1 x2 x")
V = sym.Matrix([
    [1, x1, x1**2, x1**3],
    [1, x2, x2**2, x2**3],
    [0, 1, 2*x1, 3*x1**2],
    [0, 1, 2*x2, 3*x2**2]])
V

sym.simplify(V.inv())

V_inv = sym.simplify(V.subs({x1:-1, x2:1}).inv())
V_inv

sym.factor(V_inv.T * sym.Matrix([1, x, x**2, x**3]))

def hermite_interp(fun, grad, x0=-1, x1=1, npts=101):
    jaco = (x1 - x0)/2
    x = np.linspace(-1, 1, npts)
    f1 = fun(x0)
    f2 = fun(x1)
    g1 = grad(x0)
    g2 = grad(x1)
    N1 = 1/4*(x - 1)**2 * (2 + x)
    N2 = 1/4*(x + 1)**2 * (2 - x)
    N3 = 1/4*(x - 1)**2 * (x + 1)
    N4 = 1/4*(x + 1)**2 * (x - 1)
    interp = N1*f1 + N2*f2 + jaco*(N3*g1 + N4*g2)
    return interp

def fun(x):
    return np.sin(2*np.pi*x)/(2*np.pi*x)


def grad(x):
    return np.cos(2*np.pi*x)/x - np.sin(2*np.pi*x)/(2*np.pi*x**2)

a = 2
b = 5
nels = 7
npts = 200
x = np.linspace(a, b, npts)
y = fun(x)
plt.plot(x, y, color="black")
xi = np.linspace(a, b, num=nels, endpoint=False)
dx = xi[1] - xi[0]
for x0 in xi:
    x1 = x0 + dx
    x = np.linspace(x0, x1, npts)
    y = hermite_interp(fun, grad, x0=x0, x1=x1, npts=npts)    
    plt.plot(x, y, linestyle="dashed")
plt.show();





from IPython.core.display import HTML
def css_styling():
    styles = open('./styles/custom_barba.css', 'r').read()
    return HTML(styles)
css_styling()



