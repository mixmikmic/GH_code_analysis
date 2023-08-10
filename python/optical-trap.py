import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pybisp as pb
get_ipython().magic('matplotlib inline')

# read in the data file sampled at 1.5 kHz over 20 seconds
datafile='/Users/rjoy/GitHub/pybisp/data/optical-trap.xlsx'
data = pd.ExcelFile(datafile).parse()
sample_path = data.values

# set some useful parameters
dt = 1.0/1500; t = np.arange(0, data.size*dt, dt)

k_B = 1.38064881313131e-17  # Boltzmann constant(N micron/Kelvin)
T = 300                     # temperature(Kelvin)

plt.plot(t, sample_path);

# MAP estimate assuming an Ornstein-Uhlenbeck process
ou = pb.ou.Inference(sample_path, dt)
L, D, K = ou.mapEstimate() 
k = K * (k_B*T) # physical value of the stiffness (N /micron)
print 'The best estimate for the stiffness is', k, 'N/muM '

# MAP estimate assuming equipartition
eqp = pb.equipartition.Inference(sample_path)
K = eqp.mapEstimate()
k = K * (k_B*T) # physical value of the stiffness (N /micron)
print 'The best estimate for the stiffness is', k, 'N/muM '

# error bars and posterior probability plots
dK = eqp.errorBar() 

# plot the logposterior in the '5 sigma' interval
# the 1, 2 and 3 sigma intervals are shaded
# the map estimate is shown with the red dot
eqp.plotLogProb(5)

# plot logposterior physical units around the '3 sigma' interval
kk = np.linspace(K - 3.0*dK, K + 3.0*dK)
plt.plot(kk*(k_B*T), eqp.logProb(kk), K*(k_B*T), eqp.logProb(K), 'ro');
plt.grid(), plt.xlabel('k (Newton/micron)'), plt.ylabel('log posterior');

# plot logposterior as a heat map with contours showing credible regions
LL, DD = np.meshgrid(np.linspace(L - 30, L + 30, 64), np.linspace(D - 0.01, D + 0.01, 64))
lp = ou.logProb(LL, DD) - ou.logProb(L, D)

c = plt.contourf(LL, DD, lp, 8, cmap=plt.cm.bone);plt.plot(L, D, 'ro')
plt.colorbar(c)

#values of chi-squared with 2 dof for 30%, 90% and 99% probability
levels = [-2.41, -4.60, -9.21] 
plt.contour(LL, DD, lp, levels, hold='on')
plt.xlabel('$\lambda$')
plt.ylabel('$D$');

