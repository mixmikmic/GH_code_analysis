get_ipython().magic('pylab inline')

import seaborn as sns

sns.set_context('poster', font_scale=1.25)

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

from LB_D2Q9 import pipe_cython as lb

from IPython import display

#### Input to the simulation in SI. ######
diameter = 1. # meters
length = 4. # meters

dr = 0.01 # lattice spacing
# How much you relax each iteration...related to viscosity
# Needs to be between .5 and 1.
omega = .9

viscosity = 10.**-6. # More viscous = easier to simulate!
print 'Desired viscosity:' , viscosity

input_velocity = 10.**-8. # meter/second

Re = (input_velocity * diameter)/viscosity
print 'Desired Re', Re

##### Derived parameters ######
# Re is set by diameter, input_velocity, and viscosity
# Adjust dr so that mach number doers not explode!
print
print '###### Simulation Parameters #########'
print

ly = int(np.ceil(diameter/dr))
lx = int(np.ceil(length/dr))
print 'ly', ly
print 'lx', lx

dt = ((dr**2)/(3*viscosity))*(omega-.5) # Seconds

print 'dt', dt

sim = lb.Pipe_Flow(lx=lx, ly=ly, dr=dr, dt=dt, omega=omega, 
                   input_velocity=input_velocity)

# U should be on the order of 0.1, 0.2 in the simulation!
print 'u_max in simulation:' , np.max(sim.u)

total_num_steps = 1000
steps_per_image= 10

num_to_record = int(np.ceil(float(total_num_steps)/steps_per_image))

plt.figure()
for i in range(num_to_record):
    velocity_magnitude = np.sqrt(sim.u**2 + sim.v**2)
    velocity_magnitude *= (sim.dr/sim.dt) # Scale appropriately
    plt.imshow(velocity_magnitude.T, cmap = cm.coolwarm)
    plt.grid(False)
    plt.clim(vmin=0, vmax=1.1*sim.input_velocity)
    if i == 0:
        plt.colorbar()
    display.clear_output(wait=True)
    display.display(plt.gcf())
    
    sim.run(steps_per_image)
plt.clf() # This way a second figure does not appear at the end...



