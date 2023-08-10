from __future__ import division, print_function
import numpy as np
from sympy import *
from sympy.plotting import plot3d
from scipy.linalg import eigh
from scipy.special import jn_zeros as Jn_zeros, jn as Jn
import matplotlib.pyplot as plt

init_session()
get_ipython().magic('matplotlib inline')
plt.style.use("seaborn-notebook")

def u_fun(r, m):
    """ Trial function. """
    c = symbols('c0:%i' % m)
    w = (1 - r**2) *sum(c[k]*r**(2*k) for k in range (0, m))
    return w, c

r = symbols('r')
m = 7
u, coef = u_fun(r, m)

T_inte = u**2
U_inte = diff(u, r)**2

display(U_inte)
display(T_inte)

U = integrate(expand(r*U_inte), (r, 0, 1))
T = integrate(expand(r*T_inte), (r, 0, 1))

K = Matrix(m, m, lambda ii, jj: diff(U, coef[ii], coef[jj]))
K

M = Matrix(m, m, lambda ii, jj: diff(T, coef[ii], coef[jj]))
M

Kn = np.array(K).astype(np.float64)
Mn = np.array(M).astype(np.float64)

vals, vecs = eigh(Kn, Mn, eigvals=(0, m-1))
np.sqrt(vals)

lam = Jn_zeros(0, m)

r_vec = np.linspace(0, 1, 60)
plt.figure(figsize=(14, 5))
for num in range(5):
    plt.subplot(1, 2, 1)
    u_num = lambdify((r), u.subs({coef[kk]: vecs[kk, num] for kk in range(m)}), "numpy")
    plt.plot(r_vec, u_num(r_vec)/u_num(0))
    plt.title("Approximated solution")
    plt.subplot(1, 2, 2)
    plt.plot(r_vec, Jn(0, lam[num]*r_vec), label=r"$m=%i$"%num)
    plt.title("Exact solution")
plt.legend(loc="best", framealpha=0);

def u_fun(r, m):
    """ Trial function. """
    c = symbols('c0:%i' % m)
    w = (1 - r**2) *sum(c[k]*r**(2*k) for k in range (0, m))
    return w, c

r = symbols('r')
m = 7
u, coef = u_fun(r, m)

u

U = -integrate(diff(r*diff(u, r), r)*u, (r, 0, 1))
T = integrate(r*u**2, (r, 0, 1))

K = Matrix(m, m, lambda ii, jj: diff(U, coef[ii], coef[jj]))
K

M = Matrix(m, m, lambda ii, jj: diff(T, coef[ii], coef[jj]))
M

Kn = np.array(K).astype(np.float64)
Mn = np.array(M).astype(np.float64)

vals, vecs = eigh(Kn, Mn, eigvals=(0, m-1))
np.sqrt(vals)

from IPython.core.display import HTML
def css_styling():
    styles = open('./styles/custom_barba.css', 'r').read()
    return HTML(styles)
css_styling()



