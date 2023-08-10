from __future__ import print_function, division

import sympy
sympy.init_printing()
get_ipython().magic('matplotlib inline')

s, G_C = sympy.symbols('s, G_C')
tau_c, phi = sympy.symbols('tau_c, phi')

desired_Y_over_Y_sp = sympy.exp(phi*s)/(tau_c*s + 1)

from ipywidgets import interact

t = sympy.Symbol('t', positive=True)
def plotresponse(theta=(0, 3.), tau_c_in=(1., 5.)):
    desired_response = sympy.inverse_laplace_transform(desired_Y_over_Y_sp.subs({phi: -theta, tau_c: tau_c_in})/s, s, t)
    p = sympy.plot(desired_response, (t, 0, 10), show=False)
    p2 = sympy.plot(1, (t, 0, 10), show=False)
    p.append(p2[0])
    p.show()
interact(plotresponse);

Gtilde = sympy.Symbol(r'\widetilde{G}')
actual_Y_over_Y_sp = Gtilde*G_C/(1 + Gtilde*G_C)

G_C_solved, = sympy.solve(desired_Y_over_Y_sp - actual_Y_over_Y_sp, G_C)
G_C_solved

denom = sympy.denom(G_C_solved)
G_C_rational = G_C_solved*denom/denom.subs(sympy.exp(phi*s), 1 + phi*s)

G_C_rational.simplify()

K_C, tau_I, tau_D = sympy.symbols('K_C, tau_I, tau_D', positive=True, nonzero=True)
PID = K_C*(1 + 1/(tau_I*s) + tau_D*s)
PID.expand().together()

K, tau_c, tau_1, tau_2, phi, theta = sympy.symbols('K, tau_c, tau_1, tau_2, phi, theta')
G = K*sympy.exp(phi*s)/((tau_1*s + 1)*(tau_2*s + 1))
G

target_G_C = G_C_rational.subs(Gtilde, G).expand().together()
numer, denom = (target_G_C - PID).as_numer_denom()
eq = sympy.poly(numer, s)

sympy.__version__

eqs = eq.coeffs()

sympy.solve(eqs, [K_C, tau_D, tau_I])

eqs

solution = {}
solution[K_C] = sympy.solve(eqs[1], K_C)[0]
solution[tau_D] = sympy.solve(eqs[0], tau_D)[0].subs(solution)
solution[tau_I] = sympy.solve(eqs[2], tau_I)[0].subs(solution).simplify()
solution

