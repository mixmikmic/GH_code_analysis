get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib inline')

#
# Standard Library Imports
#
import time
import math

#
# Plotting Imports
#
import matplotlib
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib import gridspec

#
# Numerical Mathematics Imports
#
import numpy as np
import scipy.sparse as sp
import numpy.linalg as la
from scipy import interpolate
from scipy.special import expit

#
# Symbolic Mathematics Imports
#
import sympy
import sympy.matrices
import sympy.physics
import sympy.physics.mechanics
import sympy.physics.mechanics.functions

#
# Nonlinear Optimization Imports
#
from optimize.snopt7 import SNOPT_solver

#
# Stanford Quadrotor-specific Code
#
import pathutils
pathutils.add_relative_to_current_source_file_path_to_sys_path("/Users/njoubert/Code/Frankencopter/Code/")

import flashlight.sympyutils as sympyutils

#
# Configuration
#
sympy.physics.mechanics.functions.mechanics_printing(use_latex="mathjax", latex_mode="equation")

#
# Let's build an expression and have a function to evaluate it
#

x1, x2, x3 = sympy.symbols('x_1 x_2 x_3')
expr = 2*x1 + x2**x3
print "My expression is:", expr

#
# Evaluate by Substitution
#

simpl = expr.subs([(x1, 1), (x2, 2), (x3, 2)])
print "Simplify with {x1 = 1, x2 = 2, x3 = 2} == ", simpl
print "Evaluate into float == ", simpl.evalf()
print "Evaluate with subs (recommended method) == ", expr.evalf(subs={x1: 1, x2: 2, x3: 2})

#
# Convert expression into anonymous function
#
func = sympy.lambdify([x1, x2, x3], expr, 'numpy')
print "Converted symbolic expression into lambda:", type(func)
print "Calling with args (1, 2, 2) == ", func(1, 2, 2)

#
# Now let's try out a vector
#

def add(A_expr, el):
    if not isinstance(A_expr,sympy.Matrix):
        A_square_expr = A_expr + el
    else:
        A_square_expr = sympy.Matrix.zeros(A_expr.rows,A_expr.cols)
        for r in range(A_expr.rows):
            for c in range(A_expr.cols):
                A_square_expr[r,c] = A_expr[r,c] + el
    return A_square_expr

nsamples = 10
# Construct a column vector of nsamples entries
w, w_entries = sympyutils.construct_matrix_and_entries("w", (nsamples, 1))

# Construct an expression of this column vector
energy = sympyutils.sum(sympyutils.square(add(w, -0.5)))
print "Energy expression is:", energy

# Create an anonymous function to evaluate this expression
func_e = sympy.lambdify(w, energy)

# Evaluate this expression with dummy data
vals = np.zeros((nsamples, 1)) + 1
print "Evaluated energy expression with all w = 1:", func_e(*vals)

#
# Now let's try setting up a jacobian matrix
#
obj = sympy.Matrix.zeros(1,1)
obj[0,0] = energy
obj.jacobian(w).transpose()

#
# Attempt to express Test 1 using SymPy to set up everything.
#

from optimize.snopt7 import SNOPT_solver
import numpy as np

inf = 1.0e20

snopt = SNOPT_solver()
snopt.setOption('Verbose',False)
snopt.setOption('Solution print',True)
snopt.setOption('Print file','blend_test1.out')

nsamples = 50

# 1. Set up decision variables
x, x_entries = sympyutils.construct_matrix_and_entries("w", (nsamples, 1))

# 2. Set up the bounds on x
low_x = np.array([0.0]*nsamples)
upp_x = np.array([1.0]*nsamples)

# 3. Set up the objective function
energy = sympyutils.sum(sympyutils.square(add(x, -0.5)))
func_e = sympy.lambdify(x, energy)

def blend_test1_objF(status,x,needF,needG,cu,iu,ru):
    obj = np.array([func_e(*x)])
    return status, obj

# 4. Set up bounds on F
low_F    = np.array([ -inf])
upp_F    = np.array([  inf])

# 5. Lets set up an equality constraint on one of the centerpoints
low_x[nsamples/2] = 1.0

# We first solve the problem without providing derivative info
snopt.snopta(name='blend_test1',x0=x0,xlow=low_x,xupp=upp_x,
             Flow=low_F,Fupp=upp_F,ObjRow=1,
             usrfun=blend_test1_objF)

plt.plot(snopt.x)
plt.ylim([0,1])
plt.title("Resulting optimized X values")


nsamples = 50
ndims = 6
nvars = ndims * nsamples

# Set up decision variables
X = sympy.Matrix.zeros(nsamples, ndims)
for r in range(nsamples):
    for c in range(ndims):
        X[r,c] = sympy.Symbol("w_%d,%d" % (r, c))

X0 = X.reshape(nvars,1)




