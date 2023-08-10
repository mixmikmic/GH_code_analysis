from sympy import *
init_printing()
from IPython.display import display

from fractions import Fraction

n = Symbol("n", integer=True)
y = Function("y")

f = y(n) - Fraction(295, 100)*y(n - 1) + 2*y(n - 2)
display(f)

eqn = Eq(f, 0)
soln = rsolve(eqn, y(n))
display(soln)
soln.evalf()

# Create non-homogeneous equation
eqn = Eq(f, -63.685*(1.07**n))
display(eqn)

soln = rsolve(eqn, y(n), init={y(0) : 2000, y(1) : 2200})
display(soln)
soln.evalf()

# Plot
get_ipython().magic('matplotlib inline')
plot(soln, (n, 0, 12), xlabel="year (n)", ylabel="fee")

# Evaluate fee at 10 years
print("Fee after 10 years (n=10): {}".format(soln.subs(n, 10).evalf()))

