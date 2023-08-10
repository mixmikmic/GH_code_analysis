from qspin.operators import hamiltonian
import numpy as np

# set total number of lattice sites
L=10 
# define drive function of time t (called 'func' above)
def drive(t,v):
  return v*t
v=0.01 # set ramp speed
# define the function arguments (called 'func_args' above)  
drive_args=[v]
# define operator strings
Jnn_indx=[[-1.0,i,(i+1)%L] for i in range(L)] # nearest-neighbour interaction with periodic BC
field_indx=[[-1.0,i] for i in range(L)] # on-site external field
# define static and dynamic lists
static_list=[['zz',Jnn_indx],['x',field_indx]] # $H_{stat} = \sum_{j=1}^{L} -\sigma^z_{j}\sigma^z_{j+1} - \sigma^x_j $
dynamic_list=[['x',field_indx,drive,drive_args]] # $H_{dyn}(t) = vt\sum_{j=1}^{L}\sigma^x_j $
# create Hamiltonian object
H=hamiltonian(static_list,dynamic_list,N=L,dtype=np.float64)



