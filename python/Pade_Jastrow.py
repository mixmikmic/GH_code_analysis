from sympy import *
init_printing() # For Sympy
from IPython.display import display

A, B, r = symbols('A B r')
u = A*r/(1.0 + B*r) - A/B
sym_u = Symbol('u')
display(Eq(sym_u,u))

# Symbolic derivatives
du = diff(u, r)
du = simplify(du)
sym_du = Symbol('du')/Symbol('dr')
display(Eq(sym_du, du))
ddu = simplify(diff(u, r, 2))
sym_ddu = Symbol('d^2')*Symbol('u')/Symbol('dr^2')
display(Eq(sym_ddu, ddu))

# Substitute concrete values to evaluate
vals = {A:-0.25, B: 0.1, r:1.0}
display(Eq(sym_u, u.subs(vals)))
display(Eq(sym_du, du.subs(vals)))
display(Eq(sym_ddu, ddu.subs(vals)))



