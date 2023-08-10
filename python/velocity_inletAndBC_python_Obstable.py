get_ipython().magic('pylab inline')

import seaborn as sns

sns.set_context('poster', font_scale=1.25)

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

from LB_D2Q9 import pipe as pip

from IPython import display

sim_p=pip.Pipe_Flow()
sim_p_Obst=pip.Pipe_Flow_Obstacles()

get_ipython().run_cell_magic('prun', '', 'sim_p_Obst.run(100)')

mag_cl = np.sqrt(sim_p.u**2 + sim_p.v**2)
plt.imshow(mag_cl.T, cmap=cm.coolwarm)
plt.grid(False)
#plt.clim(0, 1)
plt.colorbar()

#### Input to the simulation in SI. ######
diameter = 2. # meters
length = 4. # meters

deltaP = -.1

dr = 0.005
dt = dr**2

viscosity = 10.**-6. # More viscous = easier to simulate!

##### Derived parameters ######
print 'Desired viscosity:' , viscosity
#Re = (input_velocity * diameter)/viscosity
#print 'Desired Re', Re

# Re is set by diameter, input_velocity, and viscosity
# Adjust dr so that mach number doers not explode!
print
print '###### Simulation Parameters #########'
print

# Solve for the desired omega...make sure it's in the correct range.


ly = int(np.ceil(diameter/dr))
lx = int(np.ceil(length/dr))
print 'ly', ly
print 'lx', lx

nx = lx + 1
ny = ly + 1

omega = .5 + (3*dt*viscosity)/(dr**2)

print 'omega:' , omega

assert (omega > 0.5) and (omega < 1)

obstacle_size=.25 # meters

sim = pip.Pipe_Flow_Obstacle(deltaP = deltaP)

sim.run(100)

mag_cl = np.sqrt(sim.u**2 + sim.v**2)
plt.imshow(mag_cl.T, cmap=cm.coolwarm)
plt.grid(False)
#plt.clim(0, 1)
plt.colorbar()

sim_NewBC = pip.Pipe_Flow_PeriodicBC(deltaP = deltaP)
sim_NewBC.run(100)
mag_cl = np.sqrt(sim_NewBC.u**2 + sim_NewBC.v**2)
plt.imshow(mag_cl.T, cmap=cm.coolwarm)
plt.grid(False)
#plt.clim(0, 1)
plt.colorbar()

sim_NewBC_Inlet = pip.Pipe_Flow_PeriodicBC_VelocityInlet()
sim_NewBC_Inlet.run(100)
mag_cl = np.sqrt(sim_NewBC_Inlet.u**2 + sim_NewBC_Inlet.v**2)
plt.imshow(mag_cl.T, cmap=cm.coolwarm)
plt.grid(False)
#plt.clim(0, 1)
plt.colorbar()


sim_NewBC_Inlet = pip.Pipe_Flow_PeriodicBC_VelocityInlet()
sim_NewBC_Inlet.run(1000)
mag_cl = np.sqrt(sim_NewBC_Inlet.u**2 + sim_NewBC_Inlet.v**2)
plt.imshow(mag_cl.T, cmap=cm.coolwarm)
plt.grid(False)
#plt.clim(0, 1)
plt.colorbar()

sim_NewBC_Inlet = pip.Pipe_Flow_PeriodicBC_VelocityInlet()
sim_NewBC_Inlet.run(100)
mag_cl = np.sqrt(sim_NewBC_Inlet.u**2 + sim_NewBC_Inlet.v**2)
plt.imshow(mag_cl.T, cmap=cm.coolwarm)
plt.grid(False)
#plt.clim(0, 1)
plt.colorbar()

sim_NewBC_Inlet = pip.Pipe_Flow_PeriodicBC_VelocityInlet(u_w=0.1)
sim_NewBC_Inlet.run(100)
mag_cl = np.sqrt(sim_NewBC_Inlet.u**2 + sim_NewBC_Inlet.v**2)
plt.imshow(mag_cl.T, cmap=cm.coolwarm)
plt.grid(False)
#plt.clim(0, 1)
plt.colorbar()

sim_NewBC_Inlet = pip.Pipe_Flow_PeriodicBC_VelocityInlet()
sim_NewBC_Inlet.run(1000)
mag_cl = np.sqrt(sim_NewBC_Inlet.u**2 + sim_NewBC_Inlet.v**2)
plt.imshow(mag_cl.T, cmap=cm.coolwarm)
plt.grid(False)
#plt.clim(0, 1)
plt.colorbar()

get_ipython().magic('pylab inline')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
import seaborn as sns
sns.set_context('poster', font_scale=1.25)

from LB_D2Q9 import pipe_cython as lb

get_ipython().run_cell_magic('prun', '', 'sim_NewBC_Inlet.run(100)')

sim_NewBC_Inlet = lb.Pipe_Flow_PeriodicBC_VelocityInlet()
sim_NewBC_Inlet.run(100)
mag_cl = np.sqrt(sim_NewBC_Inlet.u**2 + sim_NewBC_Inlet.v**2)
plt.imshow(mag_cl.T, cmap=cm.coolwarm)
plt.grid(False)
plt.clim(0, 0.1)
plt.colorbar()

sim_NewBC_Inlet = lb.Pipe_Flow_PeriodicBC_VelocityInlet()
sim_NewBC_Inlet.run(100)
mag_cl = np.sqrt(sim_NewBC_Inlet.u**2 + sim_NewBC_Inlet.v**2)
plt.imshow(mag_cl.T, cmap=cm.coolwarm)
plt.grid(False)
#plt.clim(0, 1)
plt.colorbar()



