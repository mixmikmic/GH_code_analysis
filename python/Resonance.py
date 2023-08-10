#NAME: Resonance
#DESCRIPTION: Simulating resonance for a particle connected to a spring that is fixed to a wall.

from pycav.mechanics import *
get_ipython().magic('matplotlib notebook')
import matplotlib.pyplot as plt
import numpy as np
import pycav

def create_damped_sinusoidal_force(F, c, omega):
    def damped_sinusoidal_force(particle,t):
        return np.array([F*np.sin(omega*t),0,0]) - c*particle.v 
    return damped_sinusoidal_force

k = 1.
particles = []
particles.append(Particle(pos = np.array([-5.,0.,0.]),
                         v = np.array([0.,0.,0.]),
                         inv_mass = 1.,
                         color = [1., 1., 1.],
                         fixed = True))
particles.append(Particle(pos = np.array([5.,0.,0.]),
                         v = np.array([0.,0.,0.]),
                         inv_mass = 1.,
                         fixed = False))
springs = []
springs.append(Spring(particle_1 = particles [0],
                     particle_2 = particles [1],
                     k = k))

omega = 0.5
c = 0.1
F = 0.1
particles[1].applied_force = create_damped_sinusoidal_force(F,c,omega)

my_system = System(collides = False, interacts = False, visualize = False, particles = particles, springs = springs, record_amplitudes = True)
omega_history = []
x1_history = []
while omega < 3 :
    particles[1].applied_force = create_damped_sinusoidal_force(F,c,omega)
    my_system.time = 0.
    my_system.run_for(200, dt= 0.01)
    if particles[1].amplitude != 0.:
        x1_history.append(particles[1].amplitude)
        omega_history.append(omega)
    particles[1].pos = np.array([5.,0.,0.])
    particles[1].v = np.array([0.,0.,0.])
    for particle in particles:
        particle.max_point = np.array([None])
        particle.min_point = np.array([None])
    omega += 0.05

get_ipython().magic('matplotlib notebook')
plt.plot(omega_history,x1_history,'r',label = 'x1')
plt.xlabel('omega')
plt.ylabel('Amplitude')



