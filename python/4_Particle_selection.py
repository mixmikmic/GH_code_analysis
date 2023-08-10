import os, sys, tarfile, wget

def download_if_absent( dataset_name ):
    "Function that downloads and decompress a chosen dataset"
    if os.path.exists( dataset_name ) is False:
        tar_name = "%s.tar.gz" %dataset_name
        url = "https://github.com/openPMD/openPMD-example-datasets/raw/draft/%s" %tar_name
        wget.download( url, tar_name )
        with tarfile.open( tar_name ) as tar_file:
            tar_file.extractall()
        os.remove( tar_name )

download_if_absent( 'example-2d' )

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np

from opmd_viewer import OpenPMDTimeSeries
ts = OpenPMDTimeSeries('./example-2d/hdf5/')

# Plot the blue phase space with all the electrons
ts.get_particle( ['z', 'uz'], species='electrons', iteration=300, plot=True, vmax=1e12 );

# Select only particles that have uz between 0.05 and 0.2 
# and plot them as green dots, on top of the phase space
z_selected, uz_selected = ts.get_particle( ['z', 'uz'], species='electrons', 
                            iteration=300, select={'uz':[0.05, 0.2]} )
plt.plot(z_selected, uz_selected, 'g.')

# Plot the blue phase space with all the electrons
ts.get_particle( ['z', 'uz'], species='electrons', iteration=300, plot=True, vmax=1e12 );

# Select only particles that have uz between 0.05 and 0.2 AND z between 22 and 26
# and plot them as green dots, on top of the phase space
z_selected, uz_selected = ts.get_particle( ['z', 'uz'], species='electrons', 
                            iteration=300, select={'uz':[0.05, 0.2], 'z':[22,26]} )
plt.plot(z_selected, uz_selected, 'g.')

from opmd_viewer import ParticleTracker
# Select particles to be tracked, at iteration 300
pt = ParticleTracker( ts, iteration=300, select={'uz':[0.05,0.2], 'z':[22,26]}, species='electrons' )

plot_iteration = 300

# Plot the blue phase space with all the electrons
ts.get_particle( ['z', 'uz'], species='electrons', iteration=plot_iteration, plot=True, vmax=1e12 );

# Plot the tracked particles as red dots, on top of the phase space
z_selected, uz_selected = ts.get_particle( ['z', 'uz'], species='electrons', iteration=plot_iteration, select=pt )
plt.plot(z_selected, uz_selected, 'r.')

plot_iteration = 350

# Plot the blue phase space with all the electrons
ts.get_particle( ['z', 'uz'], species='electrons', iteration=plot_iteration, plot=True, vmax=1e12 );

# Plot the tracked particles as red dots, on top of the phase space
z_selected, uz_selected = ts.get_particle( ['z', 'uz'], species='electrons', iteration=plot_iteration, select=pt )
plt.plot(z_selected, uz_selected, 'r.')

pt = ParticleTracker( ts, iteration=300, select={'uz':[0.05,0.1], 'z':[22,26]}, 
                         species='electrons', preserve_particle_index=True )

N_iterations = len(ts.iterations)
N_particles = pt.N_selected

uz_trajectories = np.empty( ( N_iterations, N_particles ) )
for i in range( N_iterations ):
    uz, = ts.get_particle( ['uz'], select=pt, iteration=ts.iterations[i], species='electrons' )
    uz_trajectories[i, :] = uz[:]

plt.plot( ts.iterations, uz_trajectories[:,0], '-o' )
plt.plot( ts.iterations, uz_trajectories[:,10], '-o' )
plt.plot( ts.iterations, uz_trajectories[:,19], '-o' )

plt.xlabel('Iteration')
plt.ylabel('Longitudinal momentum uz')

