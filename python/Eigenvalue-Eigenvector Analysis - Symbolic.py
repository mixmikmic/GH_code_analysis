import numpy as np
import sympy

sympy.init_printing()

# We want our plots to be displayed inline, not in a separate window
get_ipython().run_line_magic('matplotlib', 'inline')

# Import the plotting functions 
import matplotlib.pyplot as plt

# Let's also improve the printing of NumPy arrays.
np.set_printoptions(precision=3, suppress=True)

# Define the symbols
m1, m2, k1, k2, k3 = sympy.symbols('m_1 m_2 k_1 k_2 k_3')

w, w1, w2 = sympy.symbols('omega omega_1 omega_2')

x1, x2 = sympy.symbols('x_1 x_2')

M = sympy.Matrix([[m1, 0],
                 [0,  m2]])

K = sympy.Matrix([[k1 + k2, -k2],
                 [-k2,      k2 + k3]])

# create the matrix to solve
KM = K - w**2 * M

eigenvalues = sympy.solve(KM.det(), w**2)

eigenvalues[0]

eigenvalues[1]

eigenvalues[0].subs([(m1, 1.0), (m2, 1.0), (k1, 4.0), (k2, 4.0), (k3, 4.0)])

eigenvalues[1].subs([(m1, 1.0), (m2, 1.0), (k1, 4.0), (k2, 4.0), (k3, 4.0)])

KM1 = K - eigenvalues[0] * M
KM2 = K - eigenvalues[1] * M

X = sympy.Matrix([[x1], 
                   [x2]])

eigenvect1 = sympy.solve(KM1 * X, (x1, x2))
eigenvect2 = sympy.solve(KM2 * X, (x1, x2))

eigenvect1

eigenvect2

eigenvect1[x1].subs([(m1, 1.0), (m2, 1.0), (k1, 4.0), (k2, 4.0), (k3, 4.0)])

eigenvect2[x1].subs([(m1, 1.0), (m2, 1.0), (k1, 4.0), (k2, 4.0), (k3, 4.0)])

A = sympy.Matrix([[0, 1, 0, 0],
            [-(k1 + k2) / m1, 0, k2 / m1, 0],
            [0, 0, 0, 1],
            [k2 / m2, 0, -(k2 + k3) / m2, 0]])

A.subs([(m1, 1.0), (m2, 1.0), (k1, 4.0), (k2, 4.0), (k3, 4.0)]).eigenvals()

A.subs([(m1, 1.0), (m2, 1.0), (k1, 4.0), (k2, 4.0), (k3, 4.0)]).eigenvects()

A.subs([(m1, 1.0), (m2, 1.0), (k1, 4.0), (k2, 4.0), (k3, 4.0)]).eigenvects()[1][2][0] * sympy.I

A.subs([(m1, 1.0), (m2, 1.0), (k1, 4.0), (k2, 4.0), (k3, 4.0)]).eigenvects()[3][2][0] * sympy.I

A = sympy.Matrix([[0, 0, -(k1 + k2), k2],
                  [0, 0, k2, (-k2 + k3)],
                  [-(k1 + k2), k2, 0, 0],
                  [k2, (-k2 + k3), 0, 0]])

B = sympy.Matrix([[-(k1 + k2), k2, 0, 0],
                  [k2, (-k2 + k3), 0, 0],
                  [0, 0, m1, 0],
                  [0, 0, 0, m2]])

A.subs([(m1, 1.0), (m2, 1.0), (k1, 4.0), (k2, 4.0), (k3, 4.0)]).eigenvals()

# Ignore this cell - We just update the CSS to make the notebook look a little bit better and easier to read

# Improve the notebook styling -- Run this first
from IPython.core.display import HTML
css_file = 'styling/CRAWLAB_IPythonNotebook.css'
HTML(open(css_file, "r").read())

