import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import sys
import os 
get_ipython().magic('matplotlib inline')

# Change directory to the code folder
os.chdir('..//code')

# Functions to sample the diffusion-weighted gradient directions
from dipy.core.sphere import disperse_charges, HemiSphere

# Function to reconstruct the tables with the acquisition information
from dipy.core.gradients import gradient_table

# Functions to perform simulations based on multi-compartment models
from dipy.sims.voxel import multi_tensor

# Import Dipy's procedures to process diffusion tensor
import dipy.reconst.dti as dti

# Importing procedures to fit the free water elimination DTI model
from functions import (wls_fit_tensor, nls_fit_tensor)

# Sample the spherical cordinates of 32 random diffusion-weighted
# directions.
n_pts = 32
theta = np.pi * np.random.rand(n_pts)
phi = 2 * np.pi * np.random.rand(n_pts)

# Convert direction to cartesian coordinates. For this, Dipy's
# class object HemiSphere is used. Since diffusion possess central
# symmetric, this class object also projects the direction to an 
# Hemisphere. 
hsph_initial = HemiSphere(theta=theta, phi=phi)

# By using a electrostatic potential energy algorithm, the directions
# of the HemiSphere class object are moved util them are evenly
# distributed in the Hemi-sphere
hsph_updated, potential = disperse_charges(hsph_initial, 5000)
directions = hsph_updated.vertices

# Based on the evenly sampled directions, the acquistion parameters are
# simulated. Vector bvals containts the information of the b-values
# while matrix bvecs contains all gradient directions for all b-value repetitions.
bvals = np.hstack((np.zeros(6), 500 * np.ones(n_pts), 1500 * np.ones(n_pts)))
bvecs = np.vstack((np.zeros((6, 3)), directions, directions))

# bvals and bvecs are converted according to Dipy's accepted format using
# Dipy's function gradient_table
gtab = gradient_table(bvals, bvecs)


# Simulations are runned for the SNR defined according to Hoy et al, 2014
SNR = 40

# Setting the volume fraction (VF) to 100%. 
VF = 100

# The value of free water diffusion is set to its known value 
Dwater = 3e-3

# Simulations are repeated for 5 levels of fractional anisotropy
FA = np.array([0.71, 0.])
L1 = np.array([1.6e-3, 0.8e-03])
L2 = np.array([0.5e-3, 0.8e-03])
L3 = np.array([0.3e-3, 0.8e-03])

# According to Hoy et al., simulations are repeated for 120 different
# diffusion tensor directions (and each direction repeated 100 times).
nDTdirs = 120
nrep = 100

# These directions are sampled using the same procedure used
# to evenly sample the diffusion gradient directions
theta = np.pi * np.random.rand(nDTdirs)
phi = 2 * np.pi * np.random.rand(nDTdirs)
hsph_initial = HemiSphere(theta=theta, phi=phi)
hsph_updated, potential = disperse_charges(hsph_initial, 5000)
DTdirs = hsph_updated.vertices

# Initializing a matrix to save all synthetic diffusion-weighted
# signals. Each dimension of this matrix corresponds to the number
# of simulated FA levels, free water volume fractions,
# diffusion tensor directions, and diffusion-weighted signals
# of the given gradient table
DWI_simulates = np.empty((FA.size, 1, nrep * nDTdirs, bvals.size))

for fa_i in range(FA.size):

    # selecting the diffusion eigenvalues for a given FA level
    mevals = np.array([[L1[fa_i], L2[fa_i], L3[fa_i]],
                       [Dwater, Dwater, Dwater]])

    # estimating volume fractions for both simulations
    # compartments (in this case 0 and 100)
    fractions = [100 - VF, VF]

    for di in range(nDTdirs):

        # Select a diffusion tensor direction
        d = DTdirs[di]

        # Repeat simulations for the given directions
        for s_i in np.arange(di * nrep, (di+1) * nrep):
            # Multi-compartmental simulations are done using
            # Dipy's function multi_tensor
            signal, sticks = multi_tensor(gtab, mevals,
                                          S0=100,
                                          angles=[d, (1, 0, 0)],
                                          fractions=fractions,
                                          snr=SNR)
            DWI_simulates[fa_i, 0, s_i, :] = signal
    prog = (fa_i+1.0) / FA.size * 100
    time.sleep(1)
    sys.stdout.write("\r%f%%" % prog)
    sys.stdout.flush()

t0 = time.time()
fw_params = wls_fit_tensor(gtab, DWI_simulates, Diso=Dwater,
                           mdreg=None)
dt = time.time() - t0
print("This step took %f seconds to run" % dt)

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
fig.subplots_adjust(hspace=0.3, wspace=0.4)

# Compute the tissue's diffusion tensor mean diffusivity
# using the functions mean_diffusivity of Dipy's module dti
md = dti.mean_diffusivity(fw_params[..., :3])

# Extract the water volume fraction estimates from the fitted
# model parameters
f = fw_params[..., 12]

# Defining the colors of the figure
colors = {0: 'r', 1: 'g'}

# Plot figures for both FA extreme levels (0 and 0.71)
for fa_i in range(FA.size):

    # Set histogram's number of bins
    nbins = 100

    # Plot tensor's mean diffusivity histograms
    axs[fa_i, 0].hist(md[fa_i, 0, :], nbins)
    axs[fa_i, 0].set_xlabel("Tensor's mean diffusivity ($mm^2/s$)")
    axs[fa_i, 0].set_ylabel('Absolute frequencies')

    # Plot water volume fraction histograms
    axs[fa_i, 1].hist(f[fa_i, 0, :], nbins)
    axs[fa_i, 1].set_xlabel('Free water estimates')
    axs[fa_i, 1].set_ylabel('Absolute frequencies')

    # Plot mean diffusivity as a function of f estimates
    axs[fa_i, 2].plot(f[fa_i, 0, :].ravel(), md[fa_i, 0, :].ravel(), '.')
    axs[fa_i, 2].set_xlabel('Free water estimates')
    axs[fa_i, 2].set_ylabel("Tensor's mean diffusivity ($mm^2/s$)")

# Save Figure
fig.savefig('Pure_free_water_F_and_tensor_MD_estimates.png')

# Sampling the free water volume fraction between 70% and 100%. 
VF = np.linspace(70, 100, 31)

# Initializing a matrix to save all synthetic diffusion-weighted
# signals. Each dimension of this matrix corresponds to the number
# of simulated FA levels, volume fractions, diffusion tensor
# directions, and diffusion-weighted signals of the given
# gradient table
DWI_simulates = np.empty((FA.size, VF.size, nrep * nDTdirs,
                          bvals.size))

for fa_i in range(FA.size):

    # selecting the diffusion eigenvalues for a given FA level
    mevals = np.array([[L1[fa_i], L2[fa_i], L3[fa_i]],
                       [Dwater, Dwater, Dwater]])

    for vf_i in range(VF.size):

        # estimating volume fractions for both simulations
        # compartments
        fractions = [100 - VF[vf_i], VF[vf_i]]

        for di in range(nDTdirs):

            # Select a diffusion tensor direction
            d = DTdirs[di]

            # Repeat simulations for the given directions
            for s_i in np.arange(di * nrep, (di+1) * nrep):
                # Multi-compartmental simulations are done using
                # Dipy's function multi_tensor
                signal, sticks = multi_tensor(gtab, mevals,
                                              S0=100,
                                              angles=[d, (1, 0, 0)],
                                              fractions=fractions,
                                              snr=SNR)
                DWI_simulates[fa_i, vf_i, s_i, :] = signal
        prog = (fa_i+1.0) * (vf_i+1.0) / (FA.size * VF.size) * 100
        time.sleep(1)
        sys.stdout.write("\r%f%%" % prog)
        sys.stdout.flush()

t0 = time.time()
fw_params = wls_fit_tensor(gtab, DWI_simulates, Diso=Dwater,
                           mdreg=None)
dt = time.time() - t0
print("This step took %f seconds to run" % dt)

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

# Compute the tissue's compartment mean diffusivity
# using function mean_diffusivity of Dipy's module dti
md = dti.mean_diffusivity(fw_params[..., :3])

# Set the md threshold to classify overestimated values
# of md
md_th = 1.5e-3;

# Initializing vector to save the percentage of high
# md values
p_high_md = np.empty(VF.size)

# Position of bar

for fa_i in range(FA.size):
    for vf_i in range(VF.size):

        # Select the mean diffusivity values for the given
        # water volume fraction and FA level 
        md_vector = md[fa_i, vf_i, :].ravel()
        p_high_md[vf_i] = (sum(md_vector > md_th) * 100.0) / (nrep*nDTdirs) 

    # Plot FA statistics as a function of the ground truth 
    # water volume fraction. Note that position of bars are
    # shifted by 0.5 so that centre of bars correspond to the
    # ground truth volume fractions
    axs[fa_i].bar(VF - 0.5, p_high_md)
    
    # Adjust properties of panels
    axs[fa_i].set_xlim([70, 100.5])
    axs[fa_i].set_ylim([0, 100])
    axs[fa_i].set_xlabel('Ground truth f-value')
    axs[fa_i].set_ylabel('Percentage of high MD')

# Save figure
fig.savefig('Percentage_High_MD.png')



