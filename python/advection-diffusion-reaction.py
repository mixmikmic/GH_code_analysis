from fenics import *
get_ipython().magic('matplotlib inline')

from mshr import *

channel = Rectangle(Point(0, 0), Point(2.2, 0.41))
cylinder = Circle(Point(0.2, 0.2), 0.05, 16)
domain = channel - cylinder

mesh = generate_mesh(domain, 64)
plot(mesh)

# Define function space for velocity
W = VectorFunctionSpace(mesh, 'P', 2)

# Define function space for system of concentrations
P1 = FiniteElement('P', triangle, 1)
element = MixedElement([P1, P1, P1])
V = FunctionSpace(mesh, element)

# Define test functions
v_1, v_2, v_3 = TestFunctions(V)

# Define functions for velocity and concentrations
w = Function(W)
# w = Constant((1,0))
u = Function(V)
u_n = Function(V)

# Split system functions to access components
u_1, u_2, u_3 = split(u)
u_n1, u_n2, u_n3 = split(u_n)

# Define source terms
f_1 = Expression('pow(x[0]-0.1,2)+pow(x[1]-0.1,2)<0.05*0.05 ? 0.1 : 0',
                 degree=1)
f_2 = Expression('pow(x[0]-0.1,2)+pow(x[1]-0.3,2)<0.05*0.05 ? 0.1 : 0',
                 degree=1)
f_3 = Constant(0)

T = 0.5            # final time
num_steps = 500    # number of time steps
dt = T / num_steps # time step size
eps = 0.01         # diffusion coefficient
K = 10.0           # reaction rate

# Define expressions used in variational forms
k = Constant(dt)
K = Constant(K)
eps = Constant(eps)

F = ((u_1 - u_n1) / k)*v_1*dx + dot(w, grad(u_1))*v_1*dx   + eps*dot(grad(u_1), grad(v_1))*dx + K*u_1*u_2*v_1*dx    + ((u_2 - u_n2) / k)*v_2*dx + dot(w, grad(u_2))*v_2*dx   + eps*dot(grad(u_2), grad(v_2))*dx + K*u_1*u_2*v_2*dx    + ((u_3 - u_n3) / k)*v_3*dx + dot(w, grad(u_3))*v_3*dx   + eps*dot(grad(u_3), grad(v_3))*dx - K*u_1*u_2*v_3*dx + K*u_3*v_3*dx   - f_1*v_1*dx - f_2*v_2*dx - f_3*v_3*dx

class NavierStokesSolver:
    # Define function spaces
    inflow   = 'near(x[0], 0)'
    outflow  = 'near(x[0], 2.2)'
    walls    = 'near(x[1], 0) || near(x[1], 0.41)'
    cylinder = 'on_boundary && x[0]>0.1 && x[0]<0.3 && x[1]>0.1 && x[1]<0.3'

    # Define inflow profile
    inflow_profile = ('4.0*1.5*x[1]*(0.41 - x[1]) / pow(0.41, 2)', '0')

    mu = 0.001         # dynamic viscosity
    rho = 1            # density

    f  = Constant((0, 0))
    mu = Constant(mu)

    def epsilon(u):
        return sym(nabla_grad(u))

    # Define stress tensor
    def sigma(self, u, p):
        return 2*self.mu*NavierStokesSolver.epsilon(u) - p*Identity(len(u))

    # Define symmetric gradient
    def __init__(self, mesh, dt):
        self.V = VectorFunctionSpace(mesh, 'P', 2)
        self.Q = FunctionSpace(mesh, 'P', 1)

        # Define boundary conditions
        bcu_inflow = DirichletBC(self.V, Expression(self.inflow_profile, degree=2), self.inflow)
        bcu_walls = DirichletBC(self.V, Constant((0, 0)), self.walls)
        bcu_cylinder = DirichletBC(self.V, Constant((0, 0)), self.cylinder)
        bcp_outflow = DirichletBC(self.Q, Constant(0), self.outflow)
        self.bcu = [bcu_inflow, bcu_walls, bcu_cylinder]
        self.bcp = [bcp_outflow]

        # Define trial and test functions
        self.v = TestFunction(self.V)
        self.q = TestFunction(self.Q)

        self.u = TrialFunction(self.V)
        self.p = TrialFunction(self.Q)

        # Define functions for solutions at previous and current time steps
        self.u_k = Function(self.V)
        self.u_  = Function(self.V)

        self.p_k = Function(self.Q)
        self.p_  = Function(self.Q)

        # Define expressions used in variational forms
        self.DT = Constant(dt)

        self.u_mid  = 0.5*(self.u_k + self.u)
        self.n  = FacetNormal(mesh)

        # Define variational problem for step 1
        self.F1 = self.rho*dot((self.u - self.u_k) / self.DT, self.v)*dx            + self.rho*dot(dot(self.u_k, nabla_grad(self.u_k)), self.v)*dx            + inner(self.sigma(self.u_mid, self.p_k), NavierStokesSolver.epsilon(self.v))*dx            + dot(self.p_k*self.n, self.v)*ds - dot(self.mu*nabla_grad(self.u_mid)*self.n, self.v)*ds            - dot(self.f, self.v)*dx

        self.a1 = lhs(self.F1)
        self.L1 = rhs(self.F1)
        self.A1 = assemble(self.a1)
        [bc.apply(self.A1) for bc in self.bcu]

        # Define variational problem for step 2
        self.a2 = dot(nabla_grad(self.p), nabla_grad(self.q))*dx
        self.L2 = dot(nabla_grad(self.p_k), nabla_grad(self.q))*dx - (1/self.DT)*div(self.u_)*self.q*dx
        self.A2 = assemble(self.a2)
        [bc.apply(self.A2) for bc in self.bcp]

        # Define variational problem for step 3
        self.a3 = dot(self.u, self.v)*dx
        self.L3 = dot(self.u_, self.v)*dx - self.DT*dot(nabla_grad(self.p_ - self.p_k), self.v)*dx
        self.A3 = assemble(self.a3)
        
    def advance(self):
        # Update current time
        # Step 1: Tentative velocity step
        b1 = assemble(self.L1)
        [bc.apply(b1) for bc in self.bcu]
        solve(self.A1, self.u_.vector(), b1, 'bicgstab', 'ilu')

        # Step 2: Pressure correction step
        b2 = assemble(self.L2)
        [bc.apply(b2) for bc in self.bcp]
        solve(self.A2, self.p_.vector(), b2, 'bicgstab', 'ilu')

        # Step 3: Velocity correction step
        b3 = assemble(self.L3)
        solve(self.A3, self.u_.vector(), b3, 'cg', 'sor')
        
        # Update previous solution
        self.u_k.assign(self.u_)
        self.p_k.assign(self.p_)

# Create VTK files for visualization output
vtkfile_u_1 = File('reaction_system/u_1.pvd')
vtkfile_u_2 = File('reaction_system/u_2.pvd')
vtkfile_u_3 = File('reaction_system/u_3.pvd')

# Create progress bar
progress = Progress('Time-stepping')
set_log_level(PROGRESS)

# Time-stepping
t = 0
out_interval = num_steps / 100;

nss = NavierStokesSolver(mesh, dt)

for k in range(num_steps):
    # Update current time
    t += dt

    # Advance the Navier-Stokes solver in time
    nss.advance()

    # Copy velocities from Navier-Stokes solver to a-d-r solver
    w.assign(nss.u_k)

    # Solve variational problem for time step
    solve(F == 0, u)

    if k%out_interval ==0 or k==num_steps:
        # Save solution to file (VTK)
        _u_1, _u_2, _u_3 = u.split()
        vtkfile_u_1 << (_u_1, t)
        vtkfile_u_2 << (_u_2, t)
        vtkfile_u_3 << (_u_3, t)

        print('u max: ', u.vector().array().max())
        
    # Update previous solution
    u_n.assign(u)

    # Update progress bar
    progress.update(t / T)



