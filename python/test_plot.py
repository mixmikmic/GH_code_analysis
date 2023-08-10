import numpy as np
from plotPSO import plotPSO_2D, plotPSO_1D
from optitestfuns import ackley

# Testing 2D plot

limits=([-5,5],[-5,5])

x_lo = limits[0][0]
x_up = limits[0][1]
y_lo = limits[1][0]
y_up = limits[1][1]

n_particles = 10

x_particles = np.random.uniform(x_lo, x_up, size=n_particles)
y_particles = np.random.uniform(y_lo, y_up, size=n_particles)

particle_xycoordinates = (x_particles, y_particles)

fig, ax = plotPSO_2D(ackley, limits, particle_xycoordinates)

# Testing 1D plot

limits=(-5,5)

x_lo = limits[0]
x_up = limits[1]

n_particles = 10

x_particles = np.random.uniform(x_lo, x_up, size=n_particles)
y_particles = np.random.uniform(y_lo, y_up, size=n_particles)

particle_xycoordinates = x_particles

fig, ax = plotPSO_1D(ackley, limits, particle_xycoordinates)

