from sympy import *

# This initialises pretty printing
init_printing()
from IPython.display import display

# This command makes plots appear inside the browser window
get_ipython().magic('matplotlib inline')

t, tau, v0 = symbols("t tau v0")
x = Function("x")

eqn = Eq(Derivative(x(t), t), v0*exp(-t/(tau)))
display(eqn)

x = dsolve(eqn, x(t))
display(x)

x = x.subs('C1', v0*tau)
display(x)

# Specify values for v0 and tau
x = x.subs(v0, 100)
x = x.subs(tau, 2)

# Plot position vs time
plot(x.args[1], (t, 0.0, 10.0), xlabel="time", ylabel="position");

classify_ode(eqn)

t, m, k, alpha = symbols("t m k alpha")
v = Function("v")
eqn = Eq((m/k)*Derivative(v(t), t), alpha*alpha - v(t)*v(t))
display(eqn)

classify_ode(eqn)

v = dsolve(eqn, v(t))
display(v)

print("Is v a solution to the ODE: {}".format(checkodesol(eqn, v)))

