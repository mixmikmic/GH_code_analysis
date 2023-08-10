from __future__ import print_function, division

get_ipython().magic('matplotlib inline')
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

## Boilerplate path hack to give access to full clustered_SNe package
import sys, os
if __package__ is None:
    if os.pardir not in sys.path[0]:
        file_dir = os.getcwd()
        sys.path.insert(0, os.path.join(file_dir, 
                                        os.pardir, 
                                        os.pardir))
        

from clustered_SNe.analysis.constants import m_proton, pc, yr, M_solar,                                    metallicity_solar
from clustered_SNe.analysis.parse import Overview, RunSummary,                                          Inputs, parse_into_scientific_notation
    
from clustered_SNe.analysis.database_helpers import session,                                                 Simulation,                                                 Simulation_Inputs,                                                 Simulation_Status
            
from clustered_SNe.analysis.fit_helpers import AggregatedResults
                                         

aggregated_results = AggregatedResults()

get_ipython().magic('matplotlib notebook')

metallicity_index = np.argmax(aggregated_results.metallicities_1D==metallicity_solar)

from mpl_toolkits.mplot3d import axes3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

with sns.plotting_context("poster"):
    surf = ax.plot_wireframe(np.log10(aggregated_results.masses_3D[   metallicity_index,:,:]),
                             np.log10(aggregated_results.densities_3D[metallicity_index,:,:]),
                             aggregated_results.momenta_3D[           metallicity_index,:,:],
                             rstride=1, cstride=1, linewidth=1)

    plt.xlabel("log Mass")
    plt.ylabel("log density")
    plt.show()

get_ipython().magic('matplotlib notebook')

density_index = np.argmax(aggregated_results.densities_1D==1.33 * m_proton)

from mpl_toolkits.mplot3d import axes3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

with sns.plotting_context("poster"):
    surf = ax.plot_wireframe(np.log10(aggregated_results.masses_3D[       :, density_index, :]),
                             np.log10(aggregated_results.metallicities_3D[:, density_index, :]),
                             aggregated_results.momenta_3D[               :, density_index, :],
                             rstride=1, cstride=1, linewidth=1)

    plt.xlabel("log Mass")
    plt.ylabel("log Z / Z_sun")
    plt.show()

MLE_fit = aggregated_results.get_MLE_fit()
Bayesian_fit = aggregated_results.get_Bayesian_fit()

get_ipython().magic('matplotlib inline')

metallicity_index = np.argmax(np.isclose(aggregated_results.metallicities_1D,
                                         metallicity_solar, atol=0))
metallicity = aggregated_results.metallicities_1D[metallicity_index]

with sns.plotting_context("poster", font_scale=2):
    for density in aggregated_results.densities_1D:
        aggregated_results.plot_slice(metallicity, density,
                                      with_MLE_fit=True, MLE_fit=MLE_fit,
                                      with_Bayesian_fit=True, Bayesian_fit=Bayesian_fit,
                                      verbose=True)
        plt.title("density = {0:.2e} g cm^-3".format(density))
        plt.show()

get_ipython().magic('matplotlib inline')

density_index = np.argmax(np.isclose(aggregated_results.densities_1D,
                                     1.33e-0 * m_proton, atol=0, rtol=1e-4))
density = aggregated_results.densities_1D[density_index]

with sns.plotting_context("poster", font_scale=2):
    for metallicity in aggregated_results.metallicities_1D:
        aggregated_results.plot_slice(metallicity, density,
                                      with_MLE_fit=True, MLE_fit=MLE_fit,
                                      with_Bayesian_fit=True, Bayesian_fit=Bayesian_fit,
                                      verbose=True)
        plt.title("log Z / Z_solar = {0:.1f}".format(np.log10(metallicity/metallicity_solar)))
        plt.show()

