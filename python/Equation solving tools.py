import sympy
sympy.init_printing()

x, y, z = sympy.symbols('x, y, z')

sympy.solve(x + y - z, z)

equations = [x + y - z, 
             2*x + y + z + 2,
             x - y - z + 2]
unknowns = [x, y, z]

get_ipython().run_cell_magic('timeit', '', 'sympy.solve(equations, unknowns)')

equations

A = sympy.Matrix([[1, 1, -1],
                  [2, 1, 1],
                  [1, -1, -1]])
b = sympy.Matrix([[0, -2, -2]]).T

A.solve(b)

get_ipython().run_cell_magic('time', '', 'A.solve(b)')

import numpy

A = numpy.matrix([[1, 1, -1],
                  [2, 1, 1],
                  [1, -1, -1]])
b = numpy.matrix([[0, -2, -2]]).T

numpy.linalg.solve(A, b)

get_ipython().run_cell_magic('time', '', 'numpy.linalg.solve(A, b)')

N = 100
bigA = numpy.random.random((N, N))

bigb = numpy.random.random((N,))

get_ipython().run_cell_magic('timeit', '', 'numpy.linalg.solve(bigA, bigb)')

bigsymbolicA = sympy.Matrix(bigA)

bigsymbolicA[0,0]

bigA[0,0]

bigsimbolicb = sympy.Matrix(bigb)

get_ipython().run_cell_magic('timeit', '', 'bigsymbolicA.solve(bigsimbolicb)')

x, y = sympy.symbols('x, y')

sympy.solve([x + sympy.log(y), y**2 - 1], [x, y])

unsolvable = x + sympy.cos(x) + sympy.log(x)
sympy.solve(unsolvable, x)

import scipy.optimize

unsolvable_numeric = sympy.lambdify(x, unsolvable)

unsolvable_numeric(50)

scipy.optimize.fsolve(unsolvable_numeric, 0.1)

def multiple_equations(unknowns):
    x, y = unknowns
    return [x + y - 1,
            x - y]

multiple_equations([1, 2])

get_ipython().run_cell_magic('timeit', '', 'first_guess = [1, 1]\nscipy.optimize.fsolve(multiple_equations, first_guess)')

h = sympy.Function('h')
t = sympy.Symbol('t', positive=True)

Fin = 2
K = 1
A = 1

Fout = K*h(t)

de = h(t).diff(t) - 1/A*(Fin - Fout)

sympy.dsolve(de, ivs={h(0): 1})

K = 2

def dhdt(h):
    Fout = K*h
    return 1/A*(Fin - Fout)

import scipy.integrate

ts = numpy.linspace(0, 10)

scipy.integrate.odeint(dhdt, ts, 1)



