#NAME: Kinetic Theory of Gases
#DESCRIPTION: Collisional and collisionless gases within a confined volume macroscopic properties e.g. pressure observed.

get_ipython().magic('matplotlib notebook')
from pycav.mechanics import *
from vpython import *
import numpy as np
import matplotlib.pyplot as plt

pressures = []
volumes = []

scene1 =  canvas(title='Gas in a box simulation')
graph1 = graph(x=0, y=0, 
      xtitle='steps', ytitle='Average pressure', 
      foreground=color.black, background=color.white, 
      xmax=1000, xmin=200)
dimension = 0.5
container = Container(dimension)
system = System(collides = False, interacts = False, visualize = True, record_pressure = True, canvas = scene1, container = container)
system.create_particles_in_container(number = 40, speed = 1, radius = 0.03)
while dimension <= 10: 
    system.pressure_history = []
    container.dimension = dimension
    dt = 1E-2
    f1 = gcurve(color=color.cyan)
    system.steps = 0
    while system.steps <= 1000:
        rate(150)
        system.simulate(dt)
        f1.plot(system.steps, system.pressure)
    pressures.append(system.pressure)
    volumes.append(dimension**3)
    dimension += 0.75

fitted = np.polyfit(np.log(volumes),np.log(pressures),1)
print ("log(p)=%.6flog(V)+%.6f"%(fitted[0],fitted[1]))
plt.plot(np.log(volumes),np.log(pressures))
plt.ylabel('log(P)')
plt.xlabel('log(V)')



