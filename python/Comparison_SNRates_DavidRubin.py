import numpy as np

import snsims
import healpy as hp

from astropy.cosmology import Planck15 as cosmo

zdist = snsims.PowerLawRates(rng=np.random.RandomState(1), 
                             fieldArea=9.6,
                             surveyDuration=10.,
                             zbinEdges=np.arange(0.10001, 1.1, 0.1))

# ten years
zdist.DeltaT

# The sky is >~ 40000 sq degrees ~ 4000 * LSST field of view
zdist.skyFraction * 2000 * 2

zdist.zSampleSize().sum()

# To compare wih David's rate divide by number of days (note bins go from 0.1 to 1.0 in steps of 0.1)
zdist.zSampleSize() / 3650.

# Get samples of those numbers and histogram (consistency, should not be new information) 
np.histogram(zdist.zSamples, np.arange(0.1, 1.01, 0.1))[0]/3650.

otherEstimate = np.array([0.0297949656, 0.0773033283291, 0.143372148164, 0.224245838014, 0.315868414203, 0.413859866222
, 0.513435128451, 0.609346340913, 0.696255385228])

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

fig, ax = plt.subplots()
zvals = np.arange(0.15, 0.96, 0.1)
ax.plot(zvals, zdist.zSampleSize()/ 3650., 'or', label='snsims ')
ax.plot(zvals, otherEstimate, 'bs', label='David Rubin')
ax.set_xlabel('z')
ax.set_ylabel('numbers per field per day')

zbin_edges = np.arange(0.1, 1.01, 0.1)
diff_volume = cosmo.comoving_volume(zbin_edges[1:]) - cosmo.comoving_volume(zbin_edges[:-1])
print diff_volume

fig_subs, axs = plt.subplots(3)
axs[0].plot(zvals, zdist.snRate(zvals), 'or')
axs[1].plot(zvals, diff_volume , 'or')
axs[2].plot(zvals, diff_volume * zdist.snRate(zvals)*10.0/40000. / 365.0, 'or')
axs[2].set_xlabel('z')
axs[0].set_ylabel('rate')
axs[1].set_ylabel('comoving vol')
axs[2].set_ylabel('vol X skyfrac X time')

zdist = snsims.PowerLawRates(rng=np.random.RandomState(1), 
                             fieldArea=18000.,
                             surveyDuration=10.,
                             zbinEdges=np.arange(0.010001, 0.901, 0.1))

zdist.zSampleSize().sum() /1.0e6



fig, ax  = plt.subplots()
_ = ax.hist(zdist.zSamples, bins=np.arange(0.001, 1.4, 0.05), histtype='step', lw=2., alpha=1.)

arcmin = 1.0 / 60. 

zdist = snsims.PowerLawRates(rng=np.random.RandomState(1), 
                             fieldArea=10.,
                             surveyDuration=10.,
                             zbinEdges=np.arange(0.010001, 0.901, 0.05))

np.array(map(np.float, zdist.numSN())) /5.

np.pi * (1.0 / 12.)**2

10.0 / 200.



