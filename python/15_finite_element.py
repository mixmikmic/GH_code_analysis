get_ipython().magic('matplotlib inline')
import numpy
import matplotlib.pyplot as plt

"""This demo program solves Poisson's equation in 1-D

    - d^2 u/ dx^2  = f(x)

on the unit interval with source f given by

    f(x) = 1

and homogeneous Dirichlet boundary conditions given by

    u(0)=u(1) = 0        for x = 0 or x = 1
"""

# Modified from demo_poisson.py from the FEniCS demos


# Begin demo

from dolfin import *

# Create mesh and define function space
mesh = UnitIntervalMesh(32)
V = FunctionSpace(mesh, "Lagrange", 1)

# Define Dirichlet boundary (x = 0 or x = 1)
def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(1.)
a = v.dx(0)*u.dx(0)*dx
L = f*v*dx 

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Save solution in VTK format
file = File("poisson.pvd")
file << u


# Plot solution
plot(u, interactive=True)

"""This demo program solves Poisson's equation in 1-D

    - d^2 u/ dx^2  = f(x)

on the unit interval with source f given by

    f(x) = 1

and boundary conditions given by

    u(0)= 0        for x = 0 
    du/dx = alpha for x=1
"""

# Modified from demo_poisson.py from the FEniCS demos


# Begin demo

from dolfin import *

# Create mesh and define function space
mesh = UnitIntervalMesh(32)
V = FunctionSpace(mesh, "Lagrange", 1)

# Define Dirichlet boundary (x = 0 )
def boundary(x):
    return x[0] < DOLFIN_EPS 

# Define boundary condition
# Dirichlet
u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)
# Neumann:  du/dx = \alpha
alpha=Constant(-.25)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(1.)
a = v.dx(0)*u.dx(0)*dx
L = v*f*dx + v*alpha*ds

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Save solution in VTK format
file = File("poisson.pvd")
file << u


# Plot solution
plot(u, interactive=True)

"""This demo program solves Poisson's equation

    - div alpha(x,y) grad u(x, y) = f(x, y)

on the a rectangular domain [0,2]x[0,1] with source f given by

    f(x, y) = 10*exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.02)

and boundary conditions given by

    u(x, y) = 0        for x = 0 or x = 2
alpha\grad u . n (x,y) = sin(5*x) for y = 0 or y = 1
"""

# modified from Dolfin Poisson demo_poisson.py

# Copyright (C) 2007-2011 Anders Logg
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# First added:  2007-08-16
# Last changed: 2012-11-12

# Begin demo

from dolfin import *

# Create mesh and define function space
mesh = RectangleMesh(0.,0.,2.,1.,64, 32,)
V = FunctionSpace(mesh, "Lagrange", 1)

# Define Dirichlet boundary (x = 0 or x = 2)
def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 2.0 - DOLFIN_EPS

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
alpha = Constant(1.)
#a = Expression("1. + 0.5*sin(8*x[0])*sin(4*x[1])")
f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)")
g = Expression("sin(5*x[0])")
a = inner(grad(v),alpha*grad(u))*dx
L = v*f*dx + v*g*ds

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Save solution in VTK format
file = File("poisson.pvd")
file << u


# Plot solution
plot(u, interactive=True)



