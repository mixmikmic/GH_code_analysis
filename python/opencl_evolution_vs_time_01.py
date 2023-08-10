get_ipython().magic('pylab inline')

import seaborn as sns

sns.set_context('poster', font_scale=1.25)

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

from LB_D2Q9 import pipe_cython as lb
from LB_D2Q9 import pipe_opencl as lb_cl

from IPython import display

#### Input to the simulation in SI. ######
diameter = 1. # meters
length = 4. # meters

deltaP = -0.01

dr = 0.01
dt = dr**2

viscosity = 10.**-1. # More viscous = easier to simulate!

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
print 'lx', lx
print 'ly', ly
nx = lx + 1
ny = ly + 1


omega = .5 + (3*dt*viscosity)/(dr**2)

print 'omega:' , omega

assert (omega > 0.5) and (omega < 1)

# U should be on the order of 0.1, 0.2 in the simulation!
#print 'u_max in simulation:' , np.max(sim.u)

import skimage as ski
import skimage.draw

import sys

sim = lb_cl.Pipe_Flow(lx=lx, ly=ly, dr=dr, dt=dt, omega=omega,
                  deltaP = deltaP)

total_num_steps = 300
steps_per_image= 10

num_to_record = int(np.ceil(float(total_num_steps)/steps_per_image))

plt.figure()
for i in range(num_to_record):
    
    fields = sim.get_fields_on_cpu()
    u = fields['u']
    v = fields['v']
    mag = np.sqrt(u**2 + v**2)
    rho = fields['rho']
    
    print 'max non_dim u:', np.max(u)
    print 'sum of mass' , np.sum(rho)/(sim.nx*sim.ny)
    plt.imshow(mag.T, cmap = cm.coolwarm)
    plt.grid(False)
    #plt.clim(vmin=0, vmax=1.)
    #if i == 0:
    #    plt.colorbar()
    display.clear_output(wait=True)
    display.display(plt.gcf())
    
    sim.run(steps_per_image)
plt.clf() # This way a second figure does not appear at the end...

plt.plot(u[300, :], ls='-', marker='.')
x = np.arange(0, 100)
plt.plot(-.000041*x*(x-100))

plt.semilogy(Ma_list, ls='-', marker='.')

plt.semilogy(Re_list, ls='-', marker='.')

