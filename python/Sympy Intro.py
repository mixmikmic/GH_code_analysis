# Typical import for numpy
# We will use a utility function or two for now...
import numpy as np

#Typical import for Matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Import sympy tools
from sympy import *
init_printing(use_latex=True)

# Make x a symbol
x = symbols('x')

# Let's write an expression
y = cos(x)

# Just provide the expression by itself,
# and it will be printed with LaTeX!
y

dydx = y.diff(x)

dydx

plot(dydx)

x_vals = np.linspace(-2*np.pi,2*np.pi,101)
y_vals = np.array([dydx.evalf(subs=dict(x=x_val)) for x_val in x_vals])
print('The length of x is %d'%(len(x_vals)))
print('The length of y is %d'%(len(y_vals)))

plt.plot(x_vals,y_vals)
plt.title('$y=%s$'%(latex(dydx)))
plt.xlabel('x')
plt.ylabel('y')
plt.show()



