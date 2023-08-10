# Warning: from fenics import * will import both `sym` and
# `q` from FEniCS. We therefore import FEniCS first and then
# overwrite these objects.
from fenics import *

def q(u):
    "Return nonlinear coefficient"
    return 1 + u**2

# Use SymPy to compute f from the manufactured solution u
import sympy as sym
x, y = sym.symbols('x[0], x[1]')
u = 1 + x + 2*y

f = - sym.diff(q(u)*sym.diff(u, x), x) - sym.diff(q(u)*sym.diff(u, y), y)
f = sym.simplify(f)

u_code = sym.printing.ccode(u)
f_code = sym.printing.ccode(f)

print('u =', u_code)
print('f =', f_code)

u_D = Expression(u_code, degree=1)

f = Expression(f_code, degree=1)

get_ipython().magic('matplotlib inline')

n = 16

mesh = UnitSquareMesh(8, 8)
plot(mesh)

V = FunctionSpace(mesh, 'P', 1)
u = Function(V)
v = TestFunction(V)

def boundary(x, on_boundary):
    return on_boundary

u_D = Expression(u_code, degree=1)
bc = DirichletBC(V, u_D, boundary)

f = Expression(f_code, degree=1)
F = q(u)*dot(grad(u), grad(v))*dx - f*v*dx

solve(F == 0, u, bc)

import numpy as np

def print_max_error():
    u_e = interpolate(u_D, V)

    u0_at_vertices = u_e.compute_vertex_values()
    u_at_vertices = u.compute_vertex_values()
    coor = V.mesh().coordinates()

    print('Max error is: ', np.max(u0_at_vertices - u_at_vertices))

def print_errors():
    u_e = interpolate(u_D, V)

    u0_at_vertices = u_e.compute_vertex_values()
    u_at_vertices = u.compute_vertex_values()
    coor = V.mesh().coordinates()

    for i, x in enumerate(coor):
        print('vertex %2d (%9g,%9g): error=%g' %
              (i, x[0], x[1],
               u0_at_vertices[i] - u_at_vertices[i]))

print_max_error()

# Compute solution
solve(F == 0, u, bc,
      solver_parameters={"newton_solver":{"relative_tolerance":1e-6}})

print_max_error()



