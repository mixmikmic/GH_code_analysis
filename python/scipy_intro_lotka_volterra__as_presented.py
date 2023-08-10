get_ipython().magic('matplotlib inline')
from collections import namedtuple
from ipywidgets import interactive
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

# some parameters we will need
x0 = 10  # initial number rabits
y0 = 2  # initial number foxes
alpha = 1  # growth rate rabits
beta = 0.1  # interaction rabit-fox
delta = 0.075  # interaction fox-rabit
gamma = 1.5  # growth rate foxes 
Coeffs = namedtuple('coeffs', ['alpha', 'beta', 'delta', 'gamma'])
coeffs = Coeffs(alpha=alpha, beta=beta, delta=delta, gamma=gamma)

get_ipython().magic('pinfo odeint')

def dX_dt(X, t, coeffs):
    alpha, beta, delta, gamma = coeffs
    x, y = X
    dx_dt = alpha * x - beta * x * y
    dy_dt = delta * x * y - gamma * y
    return np.array([dx_dt, dy_dt])

X0 = np.array([x0, y0])
t = np.linspace(0, 15, 200)
X = odeint(dX_dt, X0, t, args=(coeffs, ))
x, y = X.T
print(x[:3], y[:3])

fig, ax = plt.subplots()
ax.plot(t, x, color='green', label='x (prey, rabbits)')
ax.plot(t, y, color='blue', label='y (predator, foxes)')
ax.set_xlabel('t (time)')
ax.set_ylabel('X (population)')
ax.legend()

def solve_fox_rabbit(x0=10, y0=1/2,
                     alpha=1, beta=0.1, delta=0.075, gamma=1.5):
    
    coeffs = Coeffs(alpha=alpha, beta=beta, delta=delta, gamma=gamma)
    X0 = np.array([x0, y0])
    t = np.linspace(0, 15, 200)
    X = odeint(dX_dt, X0, t, args=(coeffs, ))
    x, y = X.T

    fig, ax = plt.subplots()
    ax.plot(t, x, color='green', label='x (prey, rabbits)')
    ax.plot(t, y, color='blue', label='y (predator, foxes)')
    ax.set_ylim([0, 50])
    ax.set_xlabel('t (time)')
    ax.set_ylabel('X (population)')
    ax.legend()

w = interactive(solve_fox_rabbit, 
                x0=(0.1, 20), y0=(0.1, 3),
                alpha=(0., 5), beta=(0.01, 2), 
                delta=(0.01, 1), gamma=(0.1, 2))
display(w)



