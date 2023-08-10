from fenics import *
get_ipython().magic('matplotlib inline')

# Create mesh and define function space
mesh = UnitSquareMesh(8, 8)
plot(mesh)

V = FunctionSpace(mesh, 'P', 1)

u = TrialFunction(V)
v = TestFunction(V)

# Define variational problem
a = dot(grad(u), grad(v))*dx

f = Constant(-6.0)
L = f*v*dx

# Define boundary condition
u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Plot solution and mesh
plot(u)
plot(mesh)

# Save solution to file in VTK format
vtkfile = File('poisson/solution.pvd')
vtkfile << u

# Compute error in L2 norm
error_L2 = errornorm(u_D, u, 'L2')

# Compute maximum error at vertices
vertex_values_u_D = u_D.compute_vertex_values(mesh)
vertex_values_u = u.compute_vertex_values(mesh)

import numpy as np
error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))

# Print errors
print('error_L2  =', error_L2)
print('error_max =', error_max)

f = Expression('x[0]>=0 && x[1]>=0 ? pow(x[0], 2) : 2', degree=2)

f = Expression('exp(-kappa*pow(pi, 2)*t)*sin(pi*k*x[0])', degree=2, kappa=1.0, t=0, k=4)

f.t += 0.1
f.k = 10

# Note that here we compare the values using a tolerance, instead of 
# compare directly with the exact value to avoid rounding errors.
tol = 1E-14
def left_boundary(x, on_boundary):
    return on_boundary and abs(x[0]) < tol
    
def right_boundary(x, on_boundary):
    return on_boundary and abs(x[0] - 1) < tol

# Note that the argument on_boundary may be omitted
def top_boundary(x):
    return abs(x[1] -1 ) < tol
    
def bottom_boundary(x):
    return abs(x[1]) < tol

u_L = Expression('1 + 2*x[1]*x[1]', degree=2)
bc_L = DirichletBC(V, u_L, left_boundary)

u_R = Expression('0', degree=2)
bc_R = DirichletBC(V, u_R, right_boundary)

bcs = [bc_L, bc_R]

u = Function(V)
solve(a == L, u, bcs)

plot(u)
plot(mesh)

