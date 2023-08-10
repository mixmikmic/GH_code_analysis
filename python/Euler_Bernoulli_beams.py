from __future__ import division
from sympy import *
from sympy import init_printing
x = symbols('x')
E, I = symbols('E I', positive=True)
C1, C2, C3, C4 = symbols('C1 C2 C3 C4')
w, M, q, f = symbols('w M q f', cls=Function)
EI = symbols('EI', cls=Function, nonnegative=True)
init_printing()

M_eq = -diff(M(x), x, 2) - q(x)

M_eq

M_sol = dsolve(M_eq, M(x)).rhs.subs([(C1, C3), (C2, C4)])

M_sol

w_eq = f(x) + diff(w(x),x,2)
w_eq

w_sol = dsolve(w_eq, w(x)).subs(f(x), M_sol/EI(x)).rhs

w_sol

sub_list = [(q(x), 0), (EI(x), E*I)]
w_sol1 = w_sol.subs(sub_list).doit()

L, F = symbols('L F')
# Fixed end
bc_eq1 = w_sol1.subs(x, 0)
bc_eq2 = diff(w_sol1, x).subs(x, 0)
# Free end
bc_eq3 = diff(w_sol1, x, 2).subs(x, L)
bc_eq4 = diff(w_sol1, x, 3).subs(x, L) + F/(E*I)

[bc_eq1, bc_eq2, bc_eq3, bc_eq4]

constants = solve([bc_eq1, bc_eq2, bc_eq3, bc_eq4], [C1, C2, C3, C4])
constants

w_sol1.subs(constants).simplify()

sub_list = [(q(x), 1), (EI(x), E*I)]
w_sol1 = w_sol.subs(sub_list).doit()

L = symbols('L')
# Fixed end
bc_eq1 = w_sol1.subs(x, 0)
bc_eq2 = diff(w_sol1, x).subs(x, 0)
# Free end
bc_eq3 = diff(w_sol1, x, 2).subs(x, L)
bc_eq4 = diff(w_sol1, x, 3).subs(x, L)

constants = solve([bc_eq1, bc_eq2, bc_eq3, bc_eq4], [C1, C2, C3, C4])

w_sol1.subs(constants).simplify()

sub_list = [(q(x), exp(x)), (EI(x), E*I)]
w_sol1 = w_sol.subs(sub_list).doit()

L = symbols('L')
# Fixed end
bc_eq1 = w_sol1.subs(x, 0)
bc_eq2 = diff(w_sol1, x).subs(x, 0)
# Free end
bc_eq3 = diff(w_sol1, x, 2).subs(x, L)
bc_eq4 = diff(w_sol1, x, 3).subs(x, L)

constants = solve([bc_eq1, bc_eq2, bc_eq3, bc_eq4], [C1, C2, C3, C4])

w_sol1.subs(constants).simplify()

k = symbols('k', integer=True)
C = symbols('C1:4')
D = symbols('D', cls=Function)

w = 6*(C1 + C2*x) - 1/(E*I)*(3*C3*x**2 + C4*x**3 -
                             6*Sum(D(k)*x**(k + 4)/((k + 1)*(k + 2)*(k + 3)*(k + 4)),(k, 0, oo)))

w



from IPython.core.display import HTML
def css_styling():
    styles = open('./styles/custom_barba.css', 'r').read()
    return HTML(styles)
css_styling()

