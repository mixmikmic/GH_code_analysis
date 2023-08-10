# import Sympy and start "pretty printing"
import sympy
sympy.init_printing()

# Define the sympy symbolic variables we'll need
t, w0, tau_0 = sympy.symbols(['t', 'omega_0', 'tau_0'], real=True, positive=True)

#--------- Input your function to examine here --------

# Use the sympy Piecewise function to define the square wave - This matches the one in the Figure 1 above.
y = 2 + sympy.Piecewise((1, t < sympy.pi/w0), (-1, t > sympy.pi/w0))


# Use the sympy Piecewise function to define the triangle wave
# First define F0
# F0 = sympy.symbols('F0')
# y = sympy.Piecewise((F0/2*t, t < sympy.pi/w0), (-(F0/2)*t + 2*F0, t >= sympy.pi/w0))


# Use the sympy Piecewise function to define a trapezoid function
# y = sympy.Piecewise((3*F0*w0/(2*sympy.pi)*t, t < (2*sympy.pi/(3*w0))), (F0, t < (4*sympy.pi/(3*w0))),
#                     (-3*F0*w0/(2*sympy.pi)*t + 3*F0, t > (4*sympy.pi/(3*w0))))

# define the number of terms to use in the approximation
num_terms = 7

# get the a0 term
a0 = w0 / (2*sympy.pi) * sympy.integrate(y, (t, 0, 2*sympy.pi/w0))

# Define matrices of 0s to fill the the an and bn terms
a = sympy.zeros(1, num_terms)
b = sympy.zeros(1, num_terms)

# cycle through the 1 to num_terms Fourier coefficients (a_n and b_n)
for n in range(num_terms):
    integral_cos = y * sympy.cos((n+1)*w0*t)         # define the integral "interior"
    a[n] = w0 / sympy.pi * sympy.integrate(integral_cos, (t, 0, 2*sympy.pi/w0))    # solve for a_n

    integral_sin = y * sympy.sin((n+1)*w0*t)         # define the integral "interior"
    b[n] = w0 / sympy.pi * sympy.integrate(integral_sin, (t, 0, 2*sympy.pi/w0))    # solve for b_n

# Simplify and display a0
sympy.simplify(a0)

# Simplify and display the an terms
sympy.simplify(a)

# Simplify and diplay the bn terms
sympy.simplify(b)

# Ignore this cell - We just update the CSS to make the notebook look a little bit better and easier to read

# Improve the notebook styling -- Run this first
from IPython.core.display import HTML
css_file = 'styling/CRAWLAB_IPythonNotebook.css'
HTML(open(css_file, "r").read())

