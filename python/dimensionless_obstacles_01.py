get_ipython().magic('pylab inline')

import seaborn as sns

sns.set_context('poster', font_scale=1.25)

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

from LB_D2Q9 import opencl_dimensionless as lb

from IPython import display

D = 1. # meter
rho = 1. # kg/m^3
nu = 1. # Viscosity, m^2/s

pressure_grad = -100 # Pa/m

pipe_length = 3*D # meter

# initialize cylinder
cylinder_center = [pipe_length/4, D/2]
cylinder_radius = D/10

sim = lb.Pipe_Flow_Cylinder(diameter=D, rho=rho, viscosity=nu, pressure_grad=pressure_grad, pipe_length=pipe_length,
                             N=25, time_prefactor=1.,
                             cylinder_center=cylinder_center, cylinder_radius=cylinder_radius,
                             two_d_local_size = (32, 32), three_d_local_size = (32, 32, 1))

plt.imshow(sim.obstacle_mask_host.T)
plt.grid(False)

time_to_run = 10
num_steps = int(time_to_run/sim.delta_t)
print 'Running for', num_steps

sim.run(num_steps)

fields = sim.get_nondim_fields()
plt.imshow(fields['u'].T, cmap=cm.coolwarm)

plt.grid(False)
plt.colorbar()

time_per_iteration = .1 #Between frames
num_steps = int(time_per_iteration/sim.delta_t)
print 'Running for', num_steps, 'iterations between frames'
print 'Corresponds to a non-dimensional time of ' , time_to_run


fields = sim.get_nondim_fields()
plt.imshow(fields['u'].T, cmap=cm.coolwarm)

plt.grid(False)
plt.colorbar()

plt.semilogy(Ma_list, ls='-', marker='.')

plt.semilogy(Re_list, ls='-', marker='.')

