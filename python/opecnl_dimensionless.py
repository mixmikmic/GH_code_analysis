get_ipython().magic('pylab inline')

import seaborn as sns
sns.set_context('poster', font_scale=1.25)

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

from LB_D2Q9 import pipe_cython as lb
from LB_D2Q9 import pipe_opencl as lb_cl

D = 10.**-2 # meter
rho = 1000 # kg/m^3
nu = 10.**-6. # Viscosity, m^2/s

pressure_grad = -1 # Pa/m

pipe_length = 5 # meter

# Create characteristic length and time scale

L = D
T = (16*rho*nu)/(np.abs(pressure_grad)*D)

# Get the reynolds number
Re = L**2/(nu*T)

print 'Re:' , '%.2e' % Re

# Get the pressure drop
deltaP = pipe_length * pressure_grad

# Dimensionless deltaP

dim_deltaP = (T**2/(rho*L**2))*deltaP

dim_deltaP

delta_x = 1./200
delta_t = delta_x**2
cs = 1./np.sqrt(3)

delta_rho = (dim_deltaP/cs**2)*(delta_t**2)/(delta_x**2)
print 'delta_rho:' , delta_rho

nu_lb = (delta_t/delta_x)**2*(1./Re)
print 'nu:' , nu_lb



