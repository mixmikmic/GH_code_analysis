get_ipython().magic('pylab inline')

import seaborn as sns

sns.set_context('poster', font_scale=1.25)

sns.set_style('white')

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

from LB_D2Q9.dimensionless import opencl_dim as lb_cl
from LB_D2Q9.dimensionless import cython_dim as lb_cython
from LB_D2Q9.dimensionless import python_dim as lb_python

from IPython import display

inlet_velocity = .1

D = 1. # meter
rho = 1. # kg/m^3
nu = 1. # Viscosity, m^2/s

pressure_grad = -10 # Pa/m

pipe_length = 3*D # meter

# initialize cylinder
cylinder_center = [pipe_length/4, D/2]
cylinder_radius = D/10

sim_cl = lb_cl.Pipe_Flow_Cylinder(diameter=D, rho=rho, viscosity=nu, pressure_grad=pressure_grad, pipe_length=pipe_length,
                             N=25, time_prefactor=1.,
                             cylinder_center=cylinder_center, cylinder_radius=cylinder_radius,
                             two_d_local_size = (32, 32), three_d_local_size = (32, 32, 1))

sim_cython = lb_cython.Velocity_Inlet_Cylinder(diameter=D, rho=rho, viscosity=nu, pipe_length=pipe_length,
                                          cylinder_center=cylinder_center, cylinder_radius=cylinder_radius,
                                          inlet_velocity=inlet_velocity,
                                          N=25, time_prefactor=1.)

sim_cython.run(100)

plt.imshow(sim_cython.rho.T, cmap=cm.coolwarm)

plt.imshow(sim_cython.u.T, cmap=cm.coolwarm)

