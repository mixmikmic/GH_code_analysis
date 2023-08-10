from fenics import *
from mshr import *
import numpy as np
get_ipython().magic('matplotlib inline')

channel = Rectangle(Point(0, 0), Point(2.2, 0.41))
cylinder = Circle(Point(0.2, 0.2), 0.05, 16)
domain = channel - cylinder

mesh = generate_mesh(domain, 64)
plot(mesh)
File('navier_stokes_cylinder/cylinder.pvd') << mesh

# Define function spaces
V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

inflow   = 'near(x[0], 0)'
outflow  = 'near(x[0], 2.2)'
walls    = 'near(x[1], 0) || near(x[1], 0.41)'
cylinder = 'on_boundary && x[0]>0.1 && x[0]<0.3 && x[1]>0.1 && x[1]<0.3'

# Define inflow profile
inflow_profile = ('4.0*1.5*x[1]*(0.41 - x[1]) / pow(0.41, 2)', '0')

# Define boundary conditions
bcu_inflow = DirichletBC(V, Expression(inflow_profile, degree=2), inflow)
bcu_walls = DirichletBC(V, Constant((0, 0)), walls)
bcu_cylinder = DirichletBC(V, Constant((0, 0)), cylinder)
bcp_outflow = DirichletBC(Q, Constant(0), outflow)
bcu = [bcu_inflow, bcu_walls, bcu_cylinder]
bcp = [bcp_outflow]

mu = 0.001         # dynamic viscosity
rho = 1            # density

T = 5.0            # final time
num_steps = 5000   # number of time steps
dt = T / num_steps # time step size

# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

# Define symmetric gradient
def epsilon(u):
    return sym(nabla_grad(u))

# Define stress tensor
def sigma(u, p):
    return 2*mu*epsilon(u) - p*Identity(len(u))

# Define functions for solutions at previous and current time steps
u_k = Function(V)
u_  = Function(V)

p_k = Function(Q)
p_  = Function(Q)

# Define expressions used in variational forms
f  = Constant((0, 0))
DT = Constant(dt)
mu = Constant(mu)

u_mid  = 0.5*(u_k + u)
n  = FacetNormal(mesh)

# Define variational problem for step 1
F1 = rho*dot((u - u_k) / DT, v)*dx    + rho*dot(dot(u_k, nabla_grad(u_k)), v)*dx    + inner(sigma(u_mid, p_k), epsilon(v))*dx    + dot(p_k*n, v)*ds - dot(mu*nabla_grad(u_mid)*n, v)*ds    - dot(f, v)*dx

a1 = lhs(F1)
L1 = rhs(F1)
A1 = assemble(a1)
[bc.apply(A1) for bc in bcu]

# Define variational problem for step 2
a2 = dot(nabla_grad(p), nabla_grad(q))*dx
L2 = dot(nabla_grad(p_k), nabla_grad(q))*dx - (1/DT)*div(u_)*q*dx
A2 = assemble(a2)
[bc.apply(A2) for bc in bcp]

# Define variational problem for step 3
a3 = dot(u, v)*dx
L3 = dot(u_, v)*dx - DT*dot(nabla_grad(p_ - p_k), v)*dx
A3 = assemble(a3)

progress = Progress('Time-stepping')
set_log_level(PROGRESS)

if 'HDF5File' in globals():
    # Use XDMF files
    xdmffile_u = XDMFFile('navier_stokes_cylinder/velocity.xdmf')
    xdmffile_p = XDMFFile('navier_stokes_cylinder/pressure.xdmf')
else:
    vtkfile_u = File('navier_stokes_cylinder/velocity.pvd')
    vtkfile_p = File('navier_stokes_cylinder/pressure.pvd')

if 'HDF5File' in globals():
    # Create time series (for use in reaction_system.py)
    timeseries_u = TimeSeries('navier_stokes_cylinder/velocity_series')
    timeseries_p = TimeSeries('navier_stokes_cylinder/pressure_series')

import sys

out_interval = num_steps / 100;

# Time-stepping
t = 0
for k in range(1,num_steps+1):
    # Update current time
    t += dt

    # Step 1: Tentative velocity step
    b1 = assemble(L1)
    [bc.apply(b1) for bc in bcu]
    solve(A1, u_.vector(), b1, 'bicgstab', 'ilu')

    # Step 2: Pressure correction step
    b2 = assemble(L2)
    [bc.apply(b2) for bc in bcp]
    solve(A2, p_.vector(), b2, 'bicgstab', 'ilu')

    # Step 3: Velocity correction step
    b3 = assemble(L3)
    solve(A3, u_.vector(), b3, 'cg', 'sor')

    # Plot solution
    # plot(u_, title='Velocity')
    # plot(p_, title='Pressure')

    if k%out_interval ==0 or k==num_steps:
        print('u max:', u_.vector().array().max())

        # Save solution to files
        if 'HDF5File' in globals():
            xdmffile_u.write(u_, t)
            xdmffile_p.write(p_, t)
        else:
            vtkfile_u << (u_, t)
            vtkfile_p << (p_, t)

    # Update previous solution
    u_k.assign(u_)
    p_k.assign(p_)

    # Update progress bar
    progress.update(t / T)
    
interactive()    

