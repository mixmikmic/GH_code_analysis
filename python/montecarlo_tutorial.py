import numpy as np
import matplotlib.pyplot as plt
import structcol as sc
import structcol.refractive_index as ri
from structcol import montecarlo as mc
from structcol import model

# For Jupyter notebooks only:
get_ipython().magic('matplotlib inline')

seed = 1

# Properties of system
ntrajectories = 100                     # number of trajectories
nevents = 100                           # number of scattering events in each trajectory
wavelen = sc.Quantity('600 nm')
radius = sc.Quantity('0.125 um')
volume_fraction = sc.Quantity(0.5, '')
n_particle = sc.Quantity(1.54, '')      # refractive indices can be specified as pint quantities or
n_matrix = ri.n('vacuum', wavelen)      # called from the refractive_index module. n_matrix is the 
n_medium = ri.n('vacuum', wavelen)      # space within sample. n_medium is outside the sample.
n_sample = ri.n_eff(n_particle, n_matrix, volume_fraction)

#%%timeit
# Calculate the phase function and scattering and absorption coefficients from the single scattering model
# (this absorption coefficient is of the scatterer, not of an absorber added to the system)
p, mu_scat, mu_abs = mc.calc_scat(radius, n_particle, n_sample, volume_fraction, wavelen, 
                                  phase_mie=False, mu_scat_mie=False)

# Initialize the trajectories
r0, k0, W0 = mc.initialize(nevents, ntrajectories, n_medium, n_sample, seed=seed, incidence_angle = 0.)
r0 = sc.Quantity(r0, 'um')
k0 = sc.Quantity(k0, '')
W0 = sc.Quantity(W0, '')

# Generate a matrix of all the randomly sampled angles first 
sintheta, costheta, sinphi, cosphi, _, _ = mc.sample_angles(nevents, ntrajectories, p)

# Create step size distribution
step = mc.sample_step(nevents, ntrajectories, mu_abs, mu_scat)
    
# Create trajectories object
trajectories = mc.Trajectory(r0, k0, W0)

# Run photons
trajectories.absorb(mu_abs, step)                         
trajectories.scatter(sintheta, costheta, sinphi, cosphi)         
trajectories.move(step)

trajectories.plot_coord(ntrajectories, three_dim=True)

z_low = sc.Quantity('0.0 um')
cutoff = sc.Quantity('50 um')

R_fraction, T_fraction = mc.calc_refl_trans(trajectories, z_low, cutoff, n_medium, n_sample)

print('R = '+ str(R_fraction))
print('T = '+ str(T_fraction))
print('Absorption coefficient = ' + str(mu_abs))

# Choose some absorption coefficient corresponding to some absorber
mu_abs_dye = sc.Quantity(0.01,'1/um')

# Create new trajectories object
trajectories_dye = mc.Trajectory(r0, k0, W0)

# Run photons
trajectories_dye.absorb(mu_abs_dye, step)                         
trajectories_dye.scatter(sintheta, costheta, sinphi, cosphi)         
trajectories_dye.move(step)
R_fraction, T_fraction = mc.calc_refl_trans(trajectories_dye, z_low, cutoff, n_medium, n_sample)

print('R = '+ str(R_fraction))
print('T = '+ str(T_fraction))
print('Absorption coefficient = ' + str(mu_abs_dye))



