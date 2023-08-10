import numpy as np 
from matplotlib import pyplot, figure
get_ipython().magic('matplotlib inline')
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 10
from time import time

# For sparse matrices
from scipy.sparse import dia_matrix
from scipy.sparse.linalg.dsolve import spsolve

from scipy.linalg import solve

import warnings
warnings.filterwarnings("ignore")

def generateMatrix(N, sigma):
    """ Computes the matrix for the diffusion equation with Crank-Nicolson
        Dirichlet condition at i=0, Neumann at i=-1
    
    Parameters:
    ----------
    N: int
        Number of discretization points
    sigma: float 
        alpha*dt/dx^2
    
    Returns:
    -------
    A: 2D numpy array of float
        Matrix for diffusion equation
    """
   
    # Setup the diagonal
    d = 2*np.diag(np.ones(N-2)*(1+1./sigma))
    
    # Consider Neumann BC
    #d[-1,-1] = 1+2./sigma
    
    # Setup upper diagonal
    ud = np.diag(np.ones(N-3)*-1, 1)
    
    # Setup lower diagonal
    ld = np.diag(np.ones(N-3)*-1, -1)
    
    A = d + ud + ld
    
    return A

def generateRHS(T, sigma):
    """ Computes right-hand side of linear system for diffusion equation
        with backward Euler
    
    Parameters:
    ----------
    T: array of float
        Temperature at current time step
    sigma: float
        alpha*dt/dx^2
    
    Returns:
    -------
    b: array of float
        Right-hand side of diffusion equation with backward Euler
    """
    
    b = T[1:-1]*2*(1./sigma-1) + T[:-2] + T[2:]
    # Consider Dirichlet BC
    b[0] += T[0]
    
    return b

def CrankNicolson(T, A, nt, sigma):
    """ Advances diffusion equation in time with Crank-Nicolson
   
    Parameters:
    ----------
    T: array of float
        initial temperature profile
    A: 2D array of float
        Matrix with discretized diffusion equation
    nt: int
        number of time steps
    sigma: float
        alpha*td/dx^2
        
    Returns:
    -------
    T: array of floats
        temperature profile after nt time steps
    """
    
    for t in range(nt):
        Tn = T.copy()
        b = generateRHS(Tn, sigma)
        # Use numpy.linalg.solve
        T_interior = solve(A,b)
        T[1:-1] = T_interior
        # Enforce Neumann BC (Dirichlet is enforced automatically)
        #T[-1] = T[-2]

    return T

# Set simulation parameters
sigma = 1.0
barX = 1.0
T = 0.1

# Set up grid parameters
nx = 100  # num of grid points
nt = 5000 # num of time steps

dx = barX / (nx - 1)   # Grid step in space
dt = T / nt            # Grid step in time

qdx = 0 # related to Neumann BCs

# Space step size
x = np.linspace(0.0, barX, nx)
# Boundary conditions
Ti = np.sin(2 * np.pi * x)

# Generate matrix
A = generateMatrix(nx, sigma)

# run implicit simulation
T = CrankNicolson(Ti.copy(), A, nt, sigma) * 100000000  # WHY???


# Initial Condition & solution plot
pyplot.figure()
pyplot.plot(x, Ti, 'b-', label='Initial Condition')
pyplot.plot(x, T, 'k-', label='Approx. Solution')
pyplot.title('Solution of the Heat Equation: Implicit', fontsize=16)
pyplot.xlabel(u'$x$', fontsize=14)
pyplot.ylabel(u'$u$', fontsize=14, rotation=0)
pyplot.legend(fontsize=14)
pyplot.show()

# Set parameters
sigma = 1.0
barX = 1.0
T = 0.1

# Set up grid parameters
nx = 10000  # num of grid points
dx = barX / (nx - 1)   # Grid step in space

# Space step size
x = np.linspace(0.0, barX, nx)
# Boundary conditions
x = np.linspace(0.0, barX, nx)
u = np.sin(2 * np.pi * x)
rhs = np.zeros(nx)

# Definition of the tridiagonal matrix
Tmatrix = [np.ones(nx), 2 *np.ones(nx), np.ones(nx)]
nonzeropositions = np.array([-1, 0, 1])
iterationMatrix = dia_matrix((Tmatrix, nonzeropositions), shape=(nx, nx))

# Solving the linear system
spSolution = spsolve(iterationMatrix, u)

# Initial Condition & solution plot
pyplot.figure()
pyplot.plot(x, u, 'b-', label='Initial Condition')
pyplot.plot(x, spSolution, 'k-', label='Approx. Solution')
pyplot.title('Solution of the Heat Equation: Explicit', fontsize=16)
pyplot.xlabel(u'$x$', fontsize=14)
pyplot.ylabel(u'$u$', fontsize=14, rotation=0)
pyplot.legend(fontsize=14)
pyplot.show()

# Stock price from x
S = np.exp(x)

# Option price from x the solution to the Heat Equation spSolution
sigma = 0.2
r = 0.07
alpha = -1/(sigma**2)*(r-sigma**2/2.)
beta = -1/(2*sigma)*(r-sigma**2/2.)**2
phi = spSolution*np.exp(alpha*x+beta*T)
V = np.exp(-r*T)*phi

# Initial Condition & solution plot
pyplot.figure()
pyplot.plot(S, V, 'b-')
pyplot.title('Solution of the BS PDE: Explicit', fontsize=16)
pyplot.xlabel(u'$x$', fontsize=14)
pyplot.ylabel(u'$u$', fontsize=14, rotation=0)
pyplot.legend(fontsize=14)
pyplot.xlim(1, 2.7)
pyplot.show()



