#Lets have matplotlib "inline"
get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

#Import packages we need
import numpy as np
from matplotlib import animation, rc
from matplotlib import pyplot as plt

#Set large figure sizes
rc('figure', figsize=(16.0, 12.0))
rc('animation', html='html5')

nx = 100
u0 = np.random.rand(nx)
u1 = np.empty(nx)
kappa = 1.0
dx = 1.0
dt = 0.8 * dx*dx / (2.0*kappa)
nt = 500

for n in range(nt):
    #Internal cells
    for i in range(1, nx-1):
        u1[i] = u0[i] + kappa*dt/(dx*dx) * (u0[i-1] - 2*u0[i] + u0[i+1])
    
    #Boundary conditions
    u1[0] = u0[0]
    u1[nx-1] = u0[nx-1]
    
    #Plot
    if (n % 100 == 0):
        plt.plot(u1, '-', label="t="+str(n*dt))
    
    #Swap u0, u1
    u0, u1 = u1, u0

plt.legend(loc='upper center', ncol=5, bbox_to_anchor=(0.5, -0.1))



