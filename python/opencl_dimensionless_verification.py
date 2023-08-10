get_ipython().magic('pylab inline')

import seaborn as sns
sns.set_context('poster', font_scale=1.25)

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

from LB_D2Q9.dimensionless import opencl_dim as lb_cl

D = 1. # meter
rho = 1. # kg/m^3
nu = 1. # Viscosity, m^2/s

pressure_grad = -10 # Pa/m

pipe_length = 3*D # meter

sim0 = lb_cl.Pipe_Flow(diameter=D, rho=rho, viscosity=nu, pressure_grad=pressure_grad, pipe_length=pipe_length,
                       N=25, time_prefactor=1.,
                       two_d_local_size = (32, 32), three_d_local_size = (32, 32, 1))

D = 1. # meter
rho = 1. # kg/m^3
nu = 1. # Viscosity, m^2/s

pressure_grad = -10 # Pa/m

pipe_length = 3*D # meter

sim1 = lb_cl.Pipe_Flow(diameter=D, rho=rho, viscosity=nu, pressure_grad=pressure_grad, pipe_length=pipe_length,
                       N=50, time_prefactor=1.,
                       two_d_local_size = (32, 32), three_d_local_size = (32, 32, 1))

D = 1. # meter
rho = 1. # kg/m^3
nu = 1. # Viscosity, m^2/s

pressure_grad = -10 # Pa/m

pipe_length = 3*D # meter

sim2 = lb_cl.Pipe_Flow(diameter=D, rho=rho, viscosity=nu, pressure_grad=pressure_grad, pipe_length=pipe_length,
                       N=150, time_prefactor=1.,
                       two_d_local_size = (32, 32), three_d_local_size = (32, 32, 1))

time_to_run = 5 # dimensionless time
num_steps = int(time_to_run/sim0.delta_t)
print 'Running for', num_steps
sim0.run(num_steps)

time_to_run = 5 # seconds

num_steps = int(time_to_run/sim1.delta_t)
print 'Running for', num_steps

sim1.run(num_steps)

num_steps = int(time_to_run/sim2.delta_t)
print 'Running for', num_steps

sim2.run(num_steps)

fields = sim0.get_fields()
plt.imshow(fields['u'].T, cmap=cm.coolwarm)
plt.grid(False)

plt.colorbar()

fields = sim1.get_fields()
plt.imshow(fields['u'].T, cmap=cm.coolwarm)
plt.grid(False)

plt.colorbar()

fields = sim2.get_fields()
plt.imshow(fields['u'].T, cmap=cm.coolwarm)
plt.grid(False)

plt.colorbar()

from mpl_toolkits.axes_grid1 import make_axes_locatable

fields = sim2.get_nondim_fields()
im = plt.imshow(fields['u'].T, cmap=cm.coolwarm)
plt.grid(False)
plt.title('Dimensionless Horizontal Velocity')

divider = make_axes_locatable(plt.gca())
cax = divider.append_axes('right', size='5%', pad=0.3)
plt.colorbar(im, cax=cax)

plt.savefig('standard_pipe_flow.png', dpi=200, bbox_inches='tight')

sns.set_style('ticks')

import matplotlib

matplotlib.rc('text', usetex=True)

fields0 = sim0.get_physical_fields()
fields1 = sim1.get_physical_fields()
fields2 = sim2.get_physical_fields()

x_values = np.arange(fields0['u'].T.shape[0])*sim0.delta_x*sim0.L
# Get the mean velocity in the x direction
mean_u = fields0['u'].T.mean(axis=1)
plt.plot(x_values, mean_u, label='N=25', ls='-', marker='.', alpha=0.5)


x_values = np.arange(fields1['u'].T.shape[0])*sim1.delta_x*sim1.L
# Get the mean velocity in the x direction
mean_u = fields1['u'].T.mean(axis=1)
plt.plot(x_values, mean_u, label='N=50', ls='-', marker='.', alpha=0.5)

x_values = np.arange(fields2['u'].T.shape[0])*sim2.delta_x*sim2.L
mean_u = fields2['u'].T.mean(axis=1)
plt.plot(x_values, mean_u, label='N=150', ls='-', marker='.', alpha=0.5)

prefactor = (1./(2*rho*nu))*pressure_grad
y = np.linspace(0, D)

predicted = prefactor*y*(y-D)
# Convert non-dim predicted

plt.plot(y, predicted, label='Theory', color='Black', ls='--')

plt.plot(y, predicted*1.25, label='Scaled Theory', color='k', ls='', marker='.', alpha=0.7)

plt.xlabel('Distance from Bottom Wall (m)')
plt.ylabel('Velocity (m/s)')

plt.title('Pipe Flow Convergence vs. Resolution (N)', y=1.04)

plt.legend(loc='best')

plt.gcf().set_size_inches(8, 6)

plt.savefig('resolution_convergence_nonscaled.png', dpi=200, bbox_inches='tight')

f1 = sim1.get_fields()

u_lb = np.max(np.abs(f1['u']))
N_lb = sim1.N
nu_lb = sim1.lb_viscosity

print (u_lb*N_lb)/nu_lb

f2 = sim2.get_fields()

u_lb = np.max(f2['u'])
N_lb = sim2.N
nu_lb = sim2.lb_viscosity

print (u_lb*N_lb)/nu_lb



