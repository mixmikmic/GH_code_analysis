get_ipython().magic('pylab inline')

import seaborn as sns

sns.set_context('poster', font_scale=1.25)

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

from LB_D2Q9 import pipe_opencl as lb_cl

from IPython import display

#### Input to the simulation in SI. ######
diameter = 4. # meters
length = 10. # meters

deltaP = -0.005

dr = 0.01
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

import skimage as ski
import skimage.draw

circle_center = np.array([2., 2.])
radius = .3

nondim_center = (circle_center/dr).astype(np.int)
nondim_radius = radius/dr

circle = ski.draw.circle(nondim_center[0], nondim_center[1], nondim_radius)

# Create the obstacle

obstacles = np.zeros((nx, ny), dtype=np.int32)
obstacles = np.asfortranarray(obstacles)
obstacles[circle[0], circle[1]] = 1
plt.imshow(obstacles.T)
plt.grid(False)

sim = lb_cl.Pipe_Flow_Obstacles(lx=lx, ly=ly, dr=dr, dt=dt, omega=omega,
                  deltaP = deltaP, obstacle_mask = obstacles)

# U should be on the order of 0.1, 0.2 in the simulation!
#print 'u_max in simulation:' , np.max(sim.u)

get_ipython().run_cell_magic('prun', '', '\nsim.run(100)')

total_num_steps = 10
steps_per_image= 1

num_to_record = int(np.ceil(float(total_num_steps)/steps_per_image))

Re_list = []
Ma_list = []

plt.figure()
for i in range(num_to_record):
    fields = sim.get_fields_on_cpu()
    u = fields['u']
    plt.imshow(u.T, cmap = cm.coolwarm)
    plt.grid(False)
    #plt.clim(vmin=0, vmax=1.)
    #if i == 0:
    #    plt.colorbar()
    display.clear_output(wait=True)
    display.display(plt.gcf())
    
    sim.run(steps_per_image)
plt.clf() # This way a second figure does not appear at the end...

plt.semilogy(Ma_list, ls='-', marker='.')

plt.semilogy(Re_list, ls='-', marker='.')

