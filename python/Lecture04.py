from sympy import *
init_printing()
from IPython.display import display

x = Symbol("x")
y = Function("y")
f = Function("f")
eqn = Eq(Derivative(y(x), x, x) + 2*Derivative(y(x), x) + y(x), 0)
display(eqn)
dsolve(eqn)

eqn = Eq(Derivative(y(x), x, x) + 2*Derivative(y(x), x) + y(x), f(x))
dsolve(eqn)

