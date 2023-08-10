from sympy import *

# This initialises pretty printing
init_printing()
from IPython.display import display

# This command makes plots appear inside the browser window
get_ipython().magic('matplotlib inline')

t, m, lmbda, k = symbols("t m lambda k")
y = Function("y")

eqn = Eq(m*Derivative(y(t), t, t) + lmbda*Derivative(y(t), t) + k*y(t), 0)
display(eqn)

print("This order of the ODE is: {}".format(ode_order(eqn, y(t))))

print("Properties of the ODE are: {}".format(classify_ode(eqn)))

y = dsolve(eqn, y(t))
display(y)

y = Function("y")
x = symbols("x")
eqn = Eq(Derivative(y(x), x, x) + 2*Derivative(y(x), x) - 3*y(x), 0)
display(eqn)

y1 = dsolve(eqn)
display(y1)

eqn = Eq(lmbda**2 + 2*lmbda -3, 0)
display(eqn)

solve(eqn)

