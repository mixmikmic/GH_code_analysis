from __future__ import division, print_function
import numpy as np
from sympy import *
from scipy.linalg import eigh
from scipy.special import jn_zeros as Jn_zeros, jn as Jn
import matplotlib.pyplot as plt

init_session()
get_ipython().magic('matplotlib inline')
plt.style.use("seaborn-notebook")

k, L = symbols('k L', positive=True)

def u_fun(x, m):
    """ Trial function. """
    c = symbols('c0:%i' % m)
    w = (1 - z)**2 * sum(c[k]*x**k for k in range (0, m))
    return w, c

m = 10
w, coef = u_fun(z, m)
display(w)

T_inte = w**2
U_inte = z*diff(w, z)**2

T = integrate(T_inte, (z, 0, 1))
U = integrate(U_inte, (z, 0, 1))

K = Matrix(m, m, lambda ii, jj: diff(U, coef[ii], coef[jj]))
M = Matrix(m, m, lambda ii, jj: diff(T, coef[ii], coef[jj]))

Kn = np.array(K).astype(np.float64)
Mn = np.array(M).astype(np.float64)

vals, vecs = eigh(Kn, Mn, eigvals=(0, m-1))
np.sqrt(vals)

lam = Jn_zeros(0, m)
lam

z_vec = np.linspace(0, 1, 100)
plt.figure(figsize=(14, 5))
for num in range(5):
    plt.subplot(1, 2, 1)
    u_num = lambdify((z), w.subs({coef[kk]: vecs[kk, num] for kk in range(m)}), "numpy")
    plt.plot(z_vec, u_num(z_vec)/u_num(0))
    plt.title("Approximated solution")
    plt.subplot(1, 2, 2)
    plt.plot(z_vec, Jn(0, lam[num]*np.sqrt(z_vec)), label=r"$m=%i$"%num)
    plt.title("Exact solution")
plt.legend(loc="best", framealpha=0);



