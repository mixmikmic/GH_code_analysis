from IPython.core.display import Image 
Image(filename='plots/asep-schema.png') 

get_ipython().magic('matplotlib inline')
import scikits.bvp_solver
import numpy as np
from matplotlib import pyplot as plt


def ASEP(a, b, N, b1, b2, epi, col):
    ''' solving the hydrodynamic approach to the ASEP problem'''
    def rhs(X, y):
        return np.array([ y[1],
                          epi*( y[1] - 2*y[0]*y[1]) 
                          ])


    def boundary_conditions(Ya, Yb):
        ''' This is to set the BC for the problem'''
        BCa = np.array([ Ya[0] - b1  ]) 
        
        BCb = np.array([ Yb[0] - b2  ]) 
        
        return BCa, BCb


    def guess_y (X):
        '''This is the guess to the answer one expects. A wise guess can help the integrator immensely!'''
        return np.array([ 
                              0.3*np.sin(X*np.pi/2), 
                              0.7*np.sin(X*np.pi/2)  ])

    problem_definition = scikits.bvp_solver.ProblemDefinition(num_ODE = 2,
                                                      num_parameters = 0,
                                                      num_left_boundary_conditions = 1,
                                                      boundary_points = (a, b),
                                                      function = rhs,
                                                      boundary_conditions = boundary_conditions)

    solution = scikits.bvp_solver.solve(bvp_problem = problem_definition,
                                solution_guess = guess_y)


    x = np.linspace(a, b, N)   # define the domain of solution 
    y = solution(x)            # solve the DE and store in y
    #Simulation completed!
    
    #Plotting business
    plt.plot(x, y[0,:],'-', color=col, linewidth=3, alpha=1, label='Low density phase')
    plt.legend(loc="upper left", fontsize=20)
    plt.xlabel("Lattice site (x)", fontsize=20)
    plt.ylabel("Density on the lattice site", fontsize=20)
    plt.title("Hydrodynamic approach to ASEP", fontsize=20)
    plt.show()
    ##########


## define parameters
a, b, N = 0.0, 1.0, 128                         # this defines the domain 
epi   = 100                                     # singularity parameters       
b1, b2  = 0.1, .45                              # BC on lattice 1
m1 = np.array([0.0])


f = plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
col = np.array(["#A60628", "blue", "black", "red", "#348ABD", ])
ASEP(a, b, N, b1, b2, epi, col[0])

import scikits.bvp_solver
import numpy as np
from matplotlib import pyplot as plt


def ASEP(a, b, N, b1, b2, epi, col):
    ''' solving the hydrodynamic approach to the ASEP problem'''
    def rhs(X, y):
        return np.array([ y[1],
                          epi*( y[1] - 2*y[0]*y[1]) 
                          ])


    def boundary_conditions(Ya, Yb):
        ''' This is to set the BC for the problem'''
        BCa = np.array([ Ya[0] - b1  ]) 
        
        BCb = np.array([ Yb[0] - b2  ]) 
        
        return BCa, BCb


    def guess_y (X):
        '''This is the guess to the answer one expects. A wise guess can help the integrator immensely!'''
        return np.array([ 
                              -0.3*np.cos(2*X*np.pi), 
                              0.7*np.cos(X*np.pi)  ])

    problem_definition = scikits.bvp_solver.ProblemDefinition(num_ODE = 2,
                                                      num_parameters = 0,
                                                      num_left_boundary_conditions = 1,
                                                      boundary_points = (a, b),
                                                      function = rhs,
                                                      boundary_conditions = boundary_conditions)

    solution = scikits.bvp_solver.solve(bvp_problem = problem_definition,
                                solution_guess = guess_y)


    x = np.linspace(a, b, N)   # define the domain of solution 
    y = solution(x)            # solve the DE and store in y
    #Simulation completed!
    
    #Plotting business
    plt.plot(x, y[0,:],'-', color=col, linewidth=3, alpha=1, label='High density phase')
    plt.legend(loc="lower left", fontsize=20)
    plt.xlabel("Lattice site (x)", fontsize=20)
    plt.ylabel("Density on the lattice site", fontsize=20)
    plt.title("Hydrodynamic approach to ASEP", fontsize=20)
##########


## define parameters
a, b, N = 0.0, 1, 128                         # this defines the domain 
epi   = 100                                     # singularity parameters       
b1, b2  = 1, .8                              # BC on lattice 1
m1 = np.array([0.0])


f = plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
col = np.array(["#A60628", "blue", "black", "red", "#348ABD", ])
ASEP(a, b, N, b1, b2, epi, col[0])

import scikits.bvp_solver
import numpy as np
from matplotlib import pyplot as plt


def ASEP(a, b, N, b1, b2, epi, col):
    ''' solving the hydrodynamic approach to the ASEP problem'''
    def rhs(X, y):
        return np.array([ y[1],
                          epi*( y[1] - 2*y[0]*y[1]) 
                          ])


    def boundary_conditions(Ya, Yb):
        ''' This is to set the BC for the problem'''
        BCa = np.array([ Ya[0] - b1  ]) 
        
        BCb = np.array([ Yb[0] - b2  ]) 
        
        return BCa, BCb


    def guess_y (X):
        '''This is the guess to the answer one expects. A wise guess can help the integrator immensely!'''
        return np.array([ 
                              -0.3*np.cos(2*X*np.pi), 
                              0.7*np.cos(X*np.pi)  ])

    problem_definition = scikits.bvp_solver.ProblemDefinition(num_ODE = 2,
                                                      num_parameters = 0,
                                                      num_left_boundary_conditions = 1,
                                                      boundary_points = (a, b),
                                                      function = rhs,
                                                      boundary_conditions = boundary_conditions)

    solution = scikits.bvp_solver.solve(bvp_problem = problem_definition,
                                solution_guess = guess_y)


    x = np.linspace(a, b, N)   # define the domain of solution 
    y = solution(x)            # solve the DE and store in y
    #Simulation completed!
    
    #Plotting business
    plt.plot(x, y[0,:],'-', color=col, linewidth=3, alpha=1, label='Maximal current phase')
    plt.legend(loc="lower left", fontsize=20)
    plt.xlabel("Lattice site (x)", fontsize=20)
    plt.ylabel("Density on the lattice site", fontsize=20)
    plt.title("Hydrodynamic approach to ASEP", fontsize=20)
##########


## define parameters
a, b, N = 0.0, 1.0, 128                         # this defines the domain 
epi   = 100                                     # singularity parameters       
b1, b2  = 1, 0                              # BC on lattice 1
m1 = np.array([0.0])


f = plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
col = np.array(["#A60628", "blue", "black", "red", "#348ABD", ])
ASEP(a, b, N, b1, b2, epi, col[0])

from IPython.core.display import Image 
Image(filename='plots/asep-schema-2lane.png') 

# Solution of y1'' = f(x) and y2''(x)=g(x) over the domain (a, b)
# We have the boundary condition y1(a), y1(b), y2(a), y2(b)
#
#
# Rajesh Singh
# 20130726
#
import scikits.bvp_solver
import numpy as np
from matplotlib import pyplot as plt


def ASEP(a, b, N, b1, b2, b3, b4, epi, m, i, col):
    ''' solving the hydrodynamic approach to the ASEP problem'''
    def rhs(X, y):
        return np.array([
                         y[1],                                                               # y1'  

                         epi*( (y[1] - 2*y[0]*y[1]) * (1-m* y[2])- y[0]*(1-y[0])*m*y[3]),     # y1'' = f(x)
                         
                         y[3],                                                               # y2'
                         
                         epi*(	(y[3] - 2*y[2]*y[3]) * (1-m* y[0])- y[2]*(1-y[2])*m*y[1])    # y2'' = g(x)
                         ]) 


    def boundary_conditions(Ya, Yb):
        ''' This is to set the BC for the problem'''
        BCa = np.array([ Ya[0] - b1, Ya[2] - b3  ]) 
        
        BCb = np.array([ Yb[0] - b2, Yb[2] - b4  ]) 
        
        return BCa, BCb


    def guess_y (X):
        '''This is the guess to the answer one expects. A wise guess can help the integrator immensely!'''
        return np.array([ 
                              0.3*np.cos(X*np.pi), 
                              0.7*np.cos(X*np.pi)  , 
                              0.01*np.cos(X*np.pi), 
                              0.01*np.cos(X*np.pi)  , 
                              ])

    problem_definition = scikits.bvp_solver.ProblemDefinition(num_ODE = 4,
                                                      num_parameters = 0,
                                                      num_left_boundary_conditions = 2,
                                                      boundary_points = (a, b),
                                                      function = rhs,
                                                      boundary_conditions = boundary_conditions)

    solution = scikits.bvp_solver.solve(bvp_problem = problem_definition,
                                solution_guess = guess_y)


    x = np.linspace(a, b, N)   # define the domain of solution 
    y = solution(x)            # solve the DE and store in y
    #Simulation completed!
    
    #Plotting business
    plt.plot(x, y[0,:],'-', color=col, linewidth=3, alpha=1, label='m=%s'%(m))
    plt.legend(loc="lower left", fontsize=20)
    plt.xlabel("Lattice site (x)", fontsize=20)
    plt.ylabel("Density on the lattice site", fontsize=20)
    plt.title("Hydrodynamic approach to ASEP", fontsize=20)
##########


## define parameters
a, b, N = 0.0, 1.0, 128                         # this defines the domain 
epi   = 100                                     # singularity parameters       
b1, b2  = 0.4, 0.2                              # BC on lattice 1
b3, b4  = 0.01, 0.8                             # BC on lattice 2
m1 = np.array([0.0, 0.65,  0.725, 0.8, 1])


f = plt.figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
col = np.array(["black", "green", "red", "#348ABD", "#A60628"])
i = 0
for i in range(5):
    ASEP(a, b, N, b1, b2, b3, b4, epi, m1[i], i, col[i])

plt.show()

