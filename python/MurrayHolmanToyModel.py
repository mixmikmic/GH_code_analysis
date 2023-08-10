from sympy import diff, lambdify, symbols, sqrt, cos
import numpy as np
import rebound
from scipy.integrate import odeint
from sympy import init_printing
init_printing()
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import FloatSlider

gamma, eta, Lambda, I, psi, phi = symbols("gamma, eta, Lambda, I, psi, phi")
H = Lambda**2/2 + gamma*I - cos(psi) - eta*cos(psi - phi)

H

var = (psi, phi, Lambda, I, gamma, eta)

Lambdadot = lambdify(var, -diff(H, psi), 'numpy')
Idot = lambdify(var, -diff(H, phi), 'numpy')
psidot = lambdify(var, diff(H, Lambda), 'numpy')
phidot = lambdify(var, diff(H, I), 'numpy')

def diffeq(y, t, *params):
    psi, phi, Lambda, I = y
    v = np.concatenate((y, params))
    dydt = [psidot(*v), phidot(*v), Lambdadot(*v), Idot(*v)]
    return dydt

def wrap(val):
    while val < -np.pi:
        val += 2*np.pi
    while val > np.pi:
        val -= 2*np.pi
    return val

npwrap = np.vectorize(wrap)

def integrate(y0, params):
    times = np.linspace(0, 100, 1000)
    sol = odeint(diffeq, y0, times, args=params)
    solpsi = npwrap(sol[:,0])
    solphi = npwrap(sol[:,1])
    solLambda = sol[:,2]
    solI = sol[:,3]
    
    fig, ax = plt.subplots(figsize=(15,10))
    #ax.set_title("Planet Mass = {0:.1e} solar masses".format(10**logmass), fontsize=24)
    ax.set_xlabel(r"$\psi$", fontsize=24)
    ax.set_ylabel(r"$\Lambda$", fontsize=24)
    
    
    ax.plot(solpsi, solLambda, '.')
    ax.set_aspect('equal')
    
    fig.show()

gamma = 2.
eta = 0.
psi0 = 0.1
phi0 = 0.
Lambda0 = 0.
I0 = 0.

y0 = (psi0, phi0, Lambda0, I0)
params = (gamma, eta)
integrate(y0, params)

def sign(y):
    return mod2pi(y[1])

def mod2pi(x):
    if x>np.pi:
        return mod2pi(x-2.*np.pi)
    if x<-np.pi:
        return mod2pi(x+2.*np.pi)
    return x

N_points_max = 2000  # maximum number of point in our Poincare Section
tmax=100
def trajectory(y0, params):          
    N_points = 0
    crossings = np.zeros((N_points_max, 4))
    
    dt = 0.1
    dt_epsilon = 0.001
    
    t=0
    y=y0
    s = sign(y)
    while t < tmax and N_points < N_points_max:
        oldt = t      
        y = odeint(diffeq, y, [oldt, oldt+dt], args=params)[1] # take the second state at oldt+dt
        t = oldt+dt
        snew = sign(y)
        if s < 0. and snew > 0.: # section crossed
            leftt = oldt
            rightt = t
            while (rightt - leftt > dt_epsilon):
                midt = (leftt+rightt)/2.
                y = odeint(diffeq, y, [t, midt], args=params)[1] # take the second state at t+dt
                t = midt
                smid = sign(y)
                if smid*s > 0.: # crossing happened after midt
                    leftt = midt
                else:
                    rightt = midt
            # Crossing found to within dt_epsilon
            crossings[N_points] = [mod2pi(y[0]), mod2pi(y[1]), y[2], y[3]]
            N_points += 1
            y = odeint(diffeq, y, [t, t+dt], args=params)[1] # move past crossing
            t = t+dt
        s = snew
    
    return crossings

gamma = 5.
eta = 0.9
params = (gamma, eta)

psi0 = 0
phi0 = 0
Lambda0 = 1
I0 = 0.01

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(15,10))
for L in np.linspace(0.2, 2, 4, endpoint=True):
    y0 = (psi0, phi0, L, I0) # populate section with trajectories with different Lambda
    crossings = trajectory(y0, params)
    ax.plot(crossings[:,0], crossings[:,2], '.b') # 0=psi, 1=phi, 2=Lambda, 3=I
    
for L in np.linspace(gamma+0.2, gamma+2, 4, endpoint=True):
    y0 = (psi0, phi0, L, I0) # populate section with trajectories with different Lambda
    crossings = trajectory(y0, params)
    ax.plot(crossings[:,0], crossings[:,2], '.b') # 0=psi, 1=phi, 2=Lambda, 3=I
    
for L in np.linspace(2.5, gamma-2, 2, endpoint=True):
    y0 = (psi0, phi0, L, I0) # populate section with trajectories with different Lambda
    crossings = trajectory(y0, params)
    ax.plot(crossings[:,0], crossings[:,2], '.b') # 0=psi, 1=phi, 2=Lambda, 3=I

ax.set_xlabel(r"$\psi$", fontsize=24)
ax.set_ylabel(r"$\Lambda$", fontsize=24)

gamma = 5.
eta = 0.9
params = (gamma, eta)

psi0 = 0
phi0 = 0
Lambda0 = 1
I0 = 0.01

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(15,10))
for I in np.linspace(0.2, 2, 4, endpoint=True):
    y0 = (psi0, phi0, L, I0) # populate section with trajectories with different Lambda
    crossings = trajectory(y0, params)
    ax.plot(crossings[:,0], crossings[:,2], '.b') # 0=psi, 1=phi, 2=Lambda, 3=I
    
for L in np.linspace(gamma+0.2, gamma+2, 4, endpoint=True):
    y0 = (psi0, phi0, L, I0) # populate section with trajectories with different Lambda
    crossings = trajectory(y0, params)
    ax.plot(crossings[:,0], crossings[:,2], '.b') # 0=psi, 1=phi, 2=Lambda, 3=I
    
for L in np.linspace(2.5, gamma-2, 2, endpoint=True):
    y0 = (psi0, phi0, L, I0) # populate section with trajectories with different Lambda
    crossings = trajectory(y0, params)
    ax.plot(crossings[:,0], crossings[:,2], '.b') # 0=psi, 1=phi, 2=Lambda, 3=I

ax.set_xlabel(r"$\psi$", fontsize=24)
ax.set_ylabel(r"$\Lambda$", fontsize=24)

