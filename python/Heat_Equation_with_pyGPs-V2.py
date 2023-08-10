import numpy as np
import sympy as sp
import pyGPs

# Linear functional L = \frac{\partial}{\partial t}*u(t, x) - \phi \frac{\partial^2}{\partial x^2}*u(t, x) was chosen. 
# Desired result: phi = 12.0.

# In general we want to arrange the t's and x's as: [[(t_0, x_0), (t_1, x_0), (t_2, x_0), ..., (t_n, x_0)][(t_0, x_1), ...], ...]
# We're setting u(t,x) = 2*x^2 + xt

# Generating data
x_u = np.linspace(0.2, 2*np.pi, 15)
t_u = np.linspace(0.2, 2*np.pi, 15)
y_u = np.exp(-t_u)*np.sin(2*np.pi*x_u)

# y_u = (2.*x_u**2 + np.outer(t_u, x_u)).T              Keeping it as simple as possible

x_f = np.linspace(0.2, 2*np.pi, 15)
t_f = np.linspace(0.2, 2*np.pi, 15)
y_f = np.exp(-t_f)*(4*np.pi**2-1)*np.sin(2*np.pi*x_f)

# y_f = x_f - 48          

# The function u is assumed to be a Gaussian Process. 
# After a linear transformation, f has to be a Gaussian Process as well.

# Need to prepare data first 
M_u = t_u.size
N_u = x_u.size
M_f = t_f.size
N_f = x_f.size

# Output data in an array
y_u.resize(M_u*N_u, 1)

# Input data in an array with two-dimensional entries
A_u = np.zeros((M_u*N_u, 2))
count = 0
for j in range(M_u):
    for i in range(N_u):
        A_u[i+j+count] = (t_u[j], x_u[i])
    count += M_u - 1

# Normally not needed
y_f = np.repeat(y_f, M_f)
y_f.resize((N_f, M_f))
    
# Output data in an array
y_f.resize(M_f*N_f, 1)

# Input data in an array with two-dimensional entries
A_f = np.zeros((M_f*N_f, 2))
count = 0
for i in range(N_f):
    for j in range(M_f):
        A_f[i+j+count] = (t_f[j], x_f[i])
    count += M_f - 1

model_u = pyGPs.GPR()
model_u.setData(A_u, y_u)
model_u.optimize(A_u, y_u)

# Note that in hyp only the logarithm of the hyperparameter is stored!
# Characteristic length-scale is equal to np.exp(hyp[0]) (Default: 1)
# Signal variance is equal to np.exp(hyp[1]) (Default: 1)

# Calculating k_ff using differentiation tools from sympy and inserting our optimal parameters.

# Declaring all the variables we need
x_i, x_j, t_i, t_j, sig_u, l_u, phi, sig_f = sp.symbols('x_i x_j t_i t_j sig_u l_u phi sig_f')
# Defining k_uu
k_uu = sig_u**2*sp.exp(-1/(2*l_u)*((x_j - x_i)**2 + (t_j - t_i)**2))
# Calculating k_ff by applying the linear transformation twice
k_ff = sp.diff(k_uu, t_j, t_i) - phi*sp.diff(k_uu, x_i, x_i, t_j) - phi*sp.diff(k_uu, t_i, x_j, x_j) + phi**2*sp.diff(k_uu, x_i, x_i, x_j, x_j)
k_ff = k_ff.subs({l_u:np.exp(model_u.covfunc.hyp[0]), sig_u:np.exp(model_u.covfunc.hyp[1])})
k_ff = sp.simplify(k_ff)
# Use this as a completely custom covariance function for pyGPs => Rather difficult

model_f = pyGPs.GPR()
model_f.setData(A_f, y_f)
model_f.setPrior(kernel = pyGPs.cov.MyKernel2()) # Custom covariance function added to the source code.
model_f.optimize()

phi = np.exp(model_f.covfunc.hyp[0])

print(phi)

# My Output: 1.0




