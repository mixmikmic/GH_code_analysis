get_ipython().magic('matplotlib inline')

from pysb import *

Model()

# Declare the monomers
Monomer('L', ['s'])
Monomer('R', ['s'])

# Declare the parameters
Parameter('L_0', 100)
Parameter('R_0', 200)
Parameter('kf', 1e-3)
Parameter('kr', 1e-3)

# Declare the initial conditions
Initial(L(s=None), L_0)
Initial(R(s=None), R_0)

# Declare the binding rule
Rule('L_binds_R', L(s=None) + R(s=None) <> L(s=1) % R(s=1), kf, kr)

# Observe the complex
Observable('LR', L(s=1) % R(s=1))

from pylab import linspace, plot, xlabel, ylabel, show
from pysb.integrate import odesolve

# Run the simulation
time = linspace(0, 40, 100)
x = odesolve(model, time)

# Plot the trajectory of LR
plot(time, x['LR'])
xlabel('Time (seconds)')
ylabel('Amount of LR')



