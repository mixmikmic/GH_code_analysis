from __future__ import print_function
from __future__ import division

get_ipython().magic('matplotlib inline')

import sympy
sympy.init_printing()

s = sympy.Symbol('s')
tau = sympy.Symbol('tau', positive=True)

G = 1/(tau*s + 2)
G

K = K_C = sympy.Symbol('K_C', positive=True)

G_OL = G*K_C

def feedback(forward, backward):
    loop = forward*backward
    return sympy.simplify(forward/(1 + loop))

G_CL = feedback(G_OL, 1)
G_CL

t = sympy.Symbol('t', positive=True)

real_CL = G_CL.subs({K_C: 1, tau: 1})
sympy.simplify(real_CL)

timeresponse = sympy.inverse_laplace_transform(sympy.simplify(real_CL/s), s, t)
timeresponse

general_timeresponse = sympy.inverse_laplace_transform(sympy.simplify(G_CL/s), s, t)
general_timeresponse

def response(new_K_C, new_tau):
    sympy.plot(general_timeresponse.subs({K_C: new_K_C, tau: new_tau}), 1, (t, 0, 4))

from ipywidgets import interact

interact(response, new_K_C=(0, 100), new_tau=(0, 20))

import matplotlib.pyplot as plt

zeta = sympy.Symbol('zeta')

G = 1/(tau**2*s**2 + 2*tau*zeta*s + 1)
G

G_CL = feedback(G*K, 1)
G_CL

def response(new_K_C, new_tau, new_zeta):
    real_CL = G_CL.subs({K_C: new_K_C, tau: new_tau, zeta: new_zeta})
    timeresponse = sympy.inverse_laplace_transform(sympy.simplify(real_CL/s), s, t)
    sympy.plot(timeresponse, 1, (t, 0, 10))
    poles = sympy.solve(sympy.denom(sympy.simplify(real_CL)), s)
    plt.plot([sympy.re(p) for p in poles], [sympy.im(p) for p in poles], 'x', markersize=10)
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.axis([-5, 5, -5, 5])

interact(response, new_K_C=(0, 100), new_tau=(0, 10.), new_zeta=(0, 2.));

