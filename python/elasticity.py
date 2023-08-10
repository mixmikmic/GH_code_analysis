from fenics import *

get_ipython().magic('matplotlib inline')

# Create mesh and define function space
length = 1; width = 0.2

mesh = BoxMesh(Point(0, 0, 0), Point(length, width, width), 10, 3, 3)
V = VectorFunctionSpace(mesh, 'P', 2)

vtkfile = File('elasticity/mesh.pvd')
vtkfile << mesh

u = TrialFunction(V)
v = TestFunction(V)

# Define strain and stress
def epsilon(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)
    #return sym(nabla_grad(u))

beta = 1.25
lambda_ = beta
mu = 1
d = u.geometric_dimension()  # space dimension

def sigma(u):
    return lambda_*nabla_div(u)*Identity(d) + 2*mu*epsilon(u)

# Define a
a = inner(sigma(u), epsilon(v))*dx

# Define L
rho = 1.
delta = width/length
gamma = 0.4*delta**2
g = gamma

#T = Constant((0,0,0))
f = Constant((0, 0, -rho*g))
L = dot(f, v)*dx # + dot(T, v)*ds

# Define boundary condition
tol = 1E-14

def clamped_boundary(x, on_boundary):
    return on_boundary and x[0] < tol

bc = DirichletBC(V, Constant((0, 0, 0)), clamped_boundary)

# Compute solution
u = Function(V)
solve(a == L, u, bc)

vtkfile = File('elasticity/solutions.pvd')
vtkfile << u

# Plot stress
s = sigma(u) - (1./3)*tr(sigma(u))*Identity(d)  # deviatoric stress
von_Mises = sqrt(3./2*inner(s, s))
V = FunctionSpace(mesh, 'P', 1)
von_Mises = project(von_Mises, V)
plot(von_Mises, title='Stress intensity')

vtkfile = File('elasticity/stress_intensity.pvd')
vtkfile << von_Mises

# Compute magnitude of displacement
u_magnitude = sqrt(dot(u, u))
u_magnitude = project(u_magnitude, V)
plot(u_magnitude, 'Displacement magnitude')
print('min/max u:',
      u_magnitude.vector().array().min(),
      u_magnitude.vector().array().max())

vtkfile = File('elasticity/disp_mag.pvd')
vtkfile << u_magnitude

