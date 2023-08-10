#NAME: 1D Velocity Distributions
#DESCRIPTION: Finding the 1D velocity distributions predicted by kinetic gas theory through simulation of hard spheres.

from pycav.mechanics import *
from vpython import *
import numpy as np
import matplotlib.pyplot as plt

container = Container(1)
system = System(collides = True, interacts = False, visualize = True, container = container)
avg_speed = 1
system.create_particles_in_container(number = 100, speed = avg_speed, radius = 0.03)
dt = 1E-2
while system.steps <= 1000:
    rate(150)
    system.simulate(dt)

get_ipython().magic('matplotlib notebook')
#alpha is a constant = 2kT/m
alpha = ((avg_speed)**2) * np.pi / (4)

@np.vectorize
def f(v):
    return ((1/(alpha*np.pi))**(0.5)) * np.exp(-(v**2)/alpha)

plt.hist(system.one_d_velocities, 20, normed = True, facecolor='green')
v = np.linspace(np.amin(system.one_d_velocities),np.amax(system.one_d_velocities),50)
l = plt.plot(v, f(v), 'r--', linewidth=1)
plt.ylabel('Probability density')
plt.xlabel('velocity')
plt.title('1D velocity distribution')
plt.show()

get_ipython().magic('matplotlib notebook')
#alpha is a constant = 2kT/m
alpha = ((avg_speed)**2) * np.pi / (4)

@np.vectorize
def f(v):
    return ((1/(alpha*np.pi))**(1.5)) * (4*np.pi*((v)**2)) * np.exp(-(v**2)/alpha)

values, bins, __ = plt.hist(system.speeds, 20, normed = True, facecolor='green')
v = np.linspace(np.amin(system.speeds),np.amax(system.speeds),20)
l = plt.plot(v, f(v), 'r--', linewidth=1)
plt.ylabel('Probability density')
plt.xlabel('velocity')
plt.title('3D speed distribution')
plt.show()





