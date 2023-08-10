import numpy as np
from halotools.sim_manager import DownloadManager, CachedHaloCatalog
#from halotools.empirical_models.abunmatch import conditional_abunmatch
from AbundanceMatching import *

halocat = CachedHaloCatalog(simname = 'bolshoi',halo_finder= 'rockstar', redshift = 0.0)

lf = np.genfromtxt('lf_r_sersic_r.dat', skip_header=True)[:,1:3]

af = AbundanceFunction(lf[:,0], lf[:,1], (-27, -5))

scatter = 0.2
remainder = af.deconvolute(scatter*LF_SCATTER_MULT, 20)

halos = np.array(halocat.halo_table)
    
nd_halos = calc_number_densities(halos['halo_vpeak'], halocat.Lbox[0])

catalog = af.match(nd_halos, scatter*LF_SCATTER_MULT)

from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')

plt.hist(catalog, bins = 100)

np.savetxt('sham_catalog.npy', catalog)



