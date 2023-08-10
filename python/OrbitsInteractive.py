#NAME: Orbits
#DESCRIPTION: Interactive (gravitational) orbit simulation.

from pycav.mechanics import *
from vpython import *
import numpy as np
from ipywidgets import widgets

def b_handler(s):
    global running
    running = True
    if running:
        dwarf_planet.pos = np.array([x.value,0.,0.])
        dwarf_planet.v = np.array([0.,0.,velocity.value])
        dwarf_planet.make_trail = False
        dwarf_planet.make_trail = True
        print(velocity)
        while True:
            rate(60)
            system.simulate(dt)


scene1 = canvas(title = "Orbits")
scene1.forward = vector(0,1,0)
giant_planet = Particle(pos = np.array([2.,0.,0.]), v = np.array([0., 0., 0.]), inv_mass = 1./200000., radius = 20, fixed = True)
dwarf_planet = Particle(pos = np.array([200.,0.,0.]), v = np.array([0., 0., 31.622]), inv_mass = 1./1., radius = 10)

dt = 0.01
planets_array = [giant_planet, dwarf_planet]
system = System(collides = False, interacts = True, visualize = True, particles = planets_array, canvas = scene1)
system.planets = planets_array

running = False

b = widgets.Button(description='Update')
display(b)
b.on_click(b_handler)

x = widgets.FloatSlider(description='Radius:', min=100, max=500, step=1, value=200)
display(x)
velocity = widgets.FloatSlider(description='Velocity:', min=20, max=100, step=1, value=31)
display(velocity)










