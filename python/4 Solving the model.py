# Setting up a custom stylesheet in IJulia
from IPython.core.display import HTML
from IPython import utils  
import urllib2
HTML(urllib2.urlopen('http://bit.ly/1Bf5Hft').read())
#HTML("""
#<style>
#open('style.css','r').read()
#</style>
#""")

get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sym
sym.init_printing() 
import solowpy

get_ipython().magic('pinfo solowpy.CobbDouglasModel.analytic_solution')

get_ipython().magic('pinfo solowpy.CobbDouglasModel.steady_state')

# define model parameters
cobb_douglas_params = {'A0': 1.0, 'L0': 1.0, 'g': 0.02, 'n': 0.03, 's': 0.15,
                      'delta': 0.05, 'alpha': 0.33}

# create an instance of the solow.Model class
cobb_douglas_model = solowpy.CobbDouglasModel(params=cobb_douglas_params)

cobb_douglas_model.steady_state

# specify some initial condition
k0 = 0.5 * cobb_douglas_model.steady_state

# grid of t values for which we want the value of k(t)
ti = np.linspace(0, 100, 10)

# generate a trajectory!
cobb_douglas_model.analytic_solution(ti, k0)

fig, ax = plt.subplots(1, 1, figsize=(8,6))

# compute the solution
ti = np.linspace(0, 100, 1000)
analytic_traj = cobb_douglas_model.analytic_solution(ti, k0)

# plot this trajectory
ax.plot(ti, analytic_traj[:,1], 'r-')

# equilibrium value of capital stock (per unit effective labor)
ax.axhline(cobb_douglas_model.steady_state, linestyle='dashed',
           color='k', label='$k^*$')

# axes, labels, title, etc
ax.set_xlabel('Time, $t$', fontsize=20, family='serif')
ax.set_ylabel('$k(t)$', rotation='horizontal', fontsize=20, family='serif')
ax.set_title('Analytic solution to a Solow model\nwith Cobb-Douglas production',
             fontsize=25, family='serif')
ax.legend(loc=0, frameon=False, bbox_to_anchor=(1.0, 1.0))
ax.grid('on')

plt.show()

# specify some initial condition
k0 = 0.5 * cobb_douglas_model.steady_state

# grid of t values for which we want the value of k(t)
ti = np.linspace(0, 100, 10)

# generate a trajectory!
cobb_douglas_model.linearized_solution(ti, k0)

# initial condition
t0, k0 = 0.0, 0.5 * cobb_douglas_model.steady_state

# grid of t values for which we want the value of k(t)
ti = np.linspace(t0, 100, 1000)

# generate the trajectories
analytic = cobb_douglas_model.analytic_solution(ti, k0)
linearized = cobb_douglas_model.linearized_solution(ti, k0)

fig, ax = plt.subplots(1, 1, figsize=(8,6))

ax.plot(ti, analytic[:,1], 'r-', label='Analytic')
ax.plot(ti, linearized[:,1], 'b-', label='Linearized')

# demarcate k*
ax.axhline(cobb_douglas_model.steady_state, linestyle='dashed', 
            color='k', label='$k^*$')

# axes, labels, title, etc
ax.set_xlabel('Time, $t$', fontsize=20, family='serif')
ax.set_ylabel('$k(t)$', rotation='horizontal', fontsize=20, family='serif')
ax.set_title('Analytic vs. linearized solutions', fontsize=25, family='serif')
ax.legend(loc='best', frameon=False, prop={'family':'serif'},
          bbox_to_anchor=(1.0, 1.0))
ax.grid('on')

fig.show()

fig, ax = plt.subplots(1, 1, figsize=(8,6))

# lower and upper bounds for initial conditions
k_star = cobb_douglas_model.steady_state
k_l = 0.5 * k_star
k_u = 2.0 * k_star

for k0 in np.linspace(k_l, k_u, 5):

    # compute the solution
    ti = np.linspace(0, 100, 1000)
    analytic_traj = solowpy.CobbDouglasModel.analytic_solution(cobb_douglas_model, ti, k0)
    
    # plot this trajectory
    ax.plot(ti, analytic_traj[:,1])

# equilibrium value of capital stock (per unit effective labor)
ax.axhline(k_star, linestyle='dashed', color='k', label='$k^*$')

# axes, labels, title, etc
ax.set_xlabel('Time, $t$', fontsize=15, family='serif')
ax.set_ylabel('$k(t)$', rotation='horizontal', fontsize=20, family='serif')
ax.set_title('Analytic solution to a Solow model\nwith Cobb-Douglas production',
             fontsize=20, family='serif')
ax.legend(loc=0, frameon=False, bbox_to_anchor=(1.0, 1.0))
ax.grid('on')

plt.show()

# define model variables
A, K, L = sym.symbols('A, K, L')

# define production parameters
alpha, sigma = sym.symbols('alpha, sigma')

# specify some production function
rho = (sigma - 1) / sigma
ces_output = (alpha * K**rho + (1 - alpha) * (A * L)**rho)**(1 / rho)

# define model parameters
ces_params = {'A0': 1.0, 'L0': 1.0, 'g': 0.02, 'n': 0.03, 's': 0.15,
              'delta': 0.05, 'alpha': 0.33, 'sigma': 0.95}

# create an instance of the solow.Model class
ces_model = solowpy.CESModel(params=ces_params)

k0 = 0.5 * ces_model.steady_state
numeric_trajectory = ces_model.ivp.solve(t0=0, y0=k0, h=0.5, T=100, integrator='dopri5')

ti = numeric_trajectory[:,0]
linearized_trajectory = ces_model.linearized_solution(ti, k0)

t0, k0 = 0.0, 0.5
numeric_soln = cobb_douglas_model.ivp.solve(t0, k0, T=100, integrator='lsoda')

fig, ax = plt.subplots(1, 1, figsize=(8,6))

# compute and plot the numeric approximation
t0, k0 = 0.0, 0.5
numeric_soln = cobb_douglas_model.ivp.solve(t0, k0, T=100, integrator='lsoda')
ax.plot(numeric_soln[:,0], numeric_soln[:,1], 'bo', markersize=3.0)

# compute and plot the analytic solution
ti = np.linspace(0, 100, 1000)
analytic_soln = solowpy.CobbDouglasModel.analytic_solution(cobb_douglas_model, ti, k0)
ax.plot(ti, analytic_soln[:,1], 'r-')

# equilibrium value of capital stock (per unit effective labor)
k_star = cobb_douglas_model.steady_state
ax.axhline(k_star, linestyle='dashed', color='k', label='$k^*$')

# axes, labels, title, etc
ax.set_xlabel('Time, $t$', fontsize=15, family='serif')
ax.set_ylabel('$k(t)$', rotation='horizontal', fontsize=20, family='serif')
ax.set_title('Numerical approximation of the solution',
             fontsize=20, family='serif')
ax.legend(loc=0, frameon=False, bbox_to_anchor=(1.0, 1.0))
ax.grid('on')

plt.show()

ti = np.linspace(0, 100, 1000)
interpolated_soln = cobb_douglas_model.ivp.interpolate(numeric_soln, ti, k=3)

fig, ax = plt.subplots(1, 1, figsize=(8,6))

# compute and plot the numeric approximation
ti = np.linspace(0, 100, 1000)
interpolated_soln = cobb_douglas_model.ivp.interpolate(numeric_soln, ti, k=3)
ax.plot(ti, interpolated_soln[:,1], 'b-')

# compute and plot the analytic solution
analytic_soln = solowpy.CobbDouglasModel.analytic_solution(cobb_douglas_model, ti, k0)
ax.plot(ti, analytic_soln[:,1], 'r-')

# equilibrium value of capital stock (per unit effective labor)
k_star = cobb_douglas_model.steady_state
ax.axhline(k_star, linestyle='dashed', color='k', label='$k^*$')

# axes, labels, title, etc
ax.set_xlabel('Time, $t$', fontsize=15, family='serif')
ax.set_ylabel('$k(t)$', rotation='horizontal', fontsize=20, family='serif')
ax.set_title('Numerical approximation of the solution',
             fontsize=20, family='serif')
ax.legend(loc=0, frameon=False, bbox_to_anchor=(1.0, 1.0))
ax.grid('on')

plt.show()



ti = np.linspace(0, 100, 1000)
residual = cobb_douglas_model.ivp.compute_residual(numeric_soln, ti, k=3)

# extract the raw residuals
capital_residual = residual[:, 1]

# typically, normalize residual by the level of the variable
norm_capital_residual = np.abs(capital_residual) / interpolated_soln[:,1]

# create the plot
fig = plt.figure(figsize=(8, 6))
plt.plot(ti, norm_capital_residual, 'b-', label='$k(t)$')
plt.axhline(np.finfo('float').eps, linestyle='dashed', color='k', label='Machine eps')
plt.xlabel('Time', fontsize=15)
plt.ylim(1e-16, 1)
plt.ylabel('Residuals (normalized)', fontsize=15, family='serif')
plt.yscale('log')
plt.title('Residual', fontsize=20, family='serif')
plt.grid()
plt.legend(loc=0, frameon=False, bbox_to_anchor=(1.0,1.0))
plt.show()



