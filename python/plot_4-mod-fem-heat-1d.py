get_ipython().magic('matplotlib inline')

import pygimli as pg
import pygimli.solver as solver
import matplotlib.pyplot as plt
import numpy as np

grid = pg.createGrid(x=np.linspace(0.0, 1.0, 100))
times = np.arange(0, 1.0, 0.04)

dirichletBC = [[1, 0],  # top
               [2, 0]]  # bottom

probeID = int(grid.nodeCount() / 2)

def uAna(t, x):
    return np.exp(-np.pi**2. * t) * np.sin(np.pi * x)

plt.plot(times, uAna(times, grid.node(probeID).pos()[0]), label='Analytical')

# u = solvePoisson(grid, times=times, theta=0.0,
#                 u0=lambda r: np.sin(np.pi * r[0]),
#                 uBoundary=dirichletBC)
dof = grid.nodeCount()
u = np.zeros((len(times), dof))
u[0, :] = list(map(lambda r: np.sin(np.pi * r[0]), grid.positions()))

dt = times[1] - times[0]
A = solver.createStiffnessMatrix(grid, np.ones(grid.cellCount()))
M = solver.createMassMatrix(grid, np.ones(grid.cellCount()))

ut = pg.RVector(dof, 0.0)
rhs = pg.RVector(dof, 0.0)
b = pg.RVector(dof, 0.0)
theta = 0

boundUdir = solver.parseArgToBoundaries(dirichletBC, grid)

for n in range(1, len(times)):
    b = (M - A * dt) * u[n - 1] + rhs * dt
    S = M

    solver.assembleDirichletBC(S, boundUdir, rhs=b)

#    solver.assembleBoundaryConditions(grid, S,
#                                      rhs=b,
#                                      boundArgs=dirichletBC,
#                                      assembler=solver.assembleDirichletBC)

    solve = pg.LinSolver(S)
    solve.solve(b, ut)

    u[n, :] = ut

# u = solver.solvePoisson(grid, times=times, theta=0.0,
#                 u0=lambda r: np.sin(np.pi * r[0]),
#                 uBoundary=dirichletBC)

plt.plot(times, u[:, probeID], label='Explicit Euler')


theta = 1

for n in range(1, len(times)):
    b = (M + A * (dt*(theta - 1.0))) * u[n-1] +         rhs * (dt*(1.0 - theta)) +         rhs * dt * theta

    b = M * u[n-1] + rhs * dt

    S = M + A * dt

    solver.assembleDirichletBC(S, boundUdir, rhs=b)

    solve = pg.LinSolver(S)
    solve.solve(b, ut)

    u[n, :] = ut

# u = solver.solvePoisson(grid, times=times, theta=1.0,
#                 u0=lambda r: np.sin(np.pi * r[0]),
#                 uBoundary=dirichletBC)

plt.plot(times, u[:, probeID], label='Implicit Euler')

u = solver.solve(grid, times=times, theta=0.5,
                 u0=lambda r: np.sin(np.pi * r[0]),
                 uB=dirichletBC)

plt.plot(times, u[:, probeID], label='Crank-Nicolson')

plt.xlabel("t[s] at x = " + str(round(grid.node(probeID).pos()[0], 2)))
plt.ylabel("u")
plt.ylim(0.0, 1.0)
plt.xlim(0.0, 0.5)
plt.legend()
plt.grid()

plt.show()

