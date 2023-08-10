

from __future__ import division
import numpy as np
from sympy import *
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib notebook')

init_printing()
u = Function('u')
t, eps= symbols('t epsilon')
omega = symbols('omega', positive=True)

eq = (1 + eps*u(t)**2)*diff(u(t), t, 2) + omega**2*u(t)
eq

u0 = Function('u0')
u1 = Function('u1')
subs = [(u(t), u0(t) + eps*u1(t))]

aux = eq.subs(subs)
aux.doit().expand()

poly = Poly(aux.doit(), eps)

coefs = poly.coeffs()
coefs

sol0 = dsolve(coefs[-1], u0(t)).rhs
sol0

C1, C2, k1, k2 = symbols('C1 C2 k1 k2')

eq_aux = expand(coefs[-2].subs(u0(t), sol0)).subs([(C1, k1), (C2, k2)])
eq_aux.doit()

sol1 = dsolve(eq_aux, u1(t)).rhs
sol1

u_app = sol0 + eps*sol1

aux_eqs = [
        sol0.subs(t, 0)-1,
        sol1.subs(t, 0),
        diff(sol0, t).subs(t, 0),
        diff(sol1, t).subs(t, 0)]
aux_eqs

coef = u_app.free_symbols - eq.free_symbols

coef

subs_sol = solve(aux_eqs, coef)
subs_sol

u_app2 = u_app.subs(subs_sol[0])
u_app2

final_sol = trigsimp(u_app2).subs(omega, 2).expand()
final_sol

trigsimp(final_sol).expand()

from scipy.integrate import odeint
def fun(x, t=0, eps=0.1):
    x0, x1 = x
    return [x1, -4*x0/(1 + eps*x0**2)]

t_vec = np.linspace(0, 100,  1000)
x = odeint(fun, [1, 0], t_vec, args=(0.1,))

lam_sol = lambdify((t, eps), final_sol, "numpy")
uu = lam_sol(t_vec, 0.1)

plt.figure(figsize=(10,8))
plt.plot(t_vec, x[:,0])
plt.plot(t_vec, uu, '--r')
plt.show()



