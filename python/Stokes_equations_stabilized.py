from fenics import *

# Load mesh and subdomains
mesh = Mesh("dolfin_fine.xml.gz")
sub_domains = MeshFunction("size_t", mesh, "dolfin_fine_subdomains.xml.gz")

# Define function spaces
P2 = VectorElement("Lagrange", mesh.ufl_cell(), 1)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = P2 * P1
W = FunctionSpace(mesh, TH)

# No-slip boundary condition for velocity
# x1 = 0, x1 = 1 and around the dolphin
noslip = Constant((0, 0))
inflow = Expression(("-sin(x[1]*pi)", "0.0"), degree=2)
zero   = Constant(0)

# No-slip boundary condition for velocity
bc0 = DirichletBC(W.sub(0), noslip, sub_domains, 0)

# Inflow boundary condition for velocity
bc1 = DirichletBC(W.sub(0), inflow, sub_domains, 1)

# Boundary condition for pressure at outflow
bc2 = DirichletBC(W.sub(1), zero, sub_domains, 2)

# Collect boundary conditions
bcs = [bc0, bc1, bc2]

# Define variational problem
(v, q) = TestFunctions(W)
(u, p) = TrialFunctions(W)
f = Constant((0, 0))
h = CellSize(mesh)
beta  = 0.2
delta = beta*h*h
a = (inner(grad(v), grad(u)) - div(v)*p - q*div(u) -     delta*inner(grad(q), grad(p)))*dx
L = inner(v - delta*grad(q), f)*dx

# Compute solution
w = Function(W)
solve(a == L, w, bcs)
u, p = w.split()

# Save solution in VTK format
ufile_pvd = File("velocity.pvd")
ufile_pvd << u
pfile_pvd = File("pressure.pvd")
pfile_pvd << p

get_ipython().magic('matplotlib inline')
# Plot solution
plot(u)

plot(p)

