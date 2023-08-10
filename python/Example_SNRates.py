import numpy as np

import snsims
import healpy as hp

from astropy.cosmology import Planck15 as cosmo

help(snsims.PowerLawRates)

zdist = snsims.PowerLawRates(rng=np.random.RandomState(1), 
                             fieldArea=9.6,
                             surveyDuration=10.,
                             zbinEdges=np.arange(0.010001, 1.1, 0.1))

# ten years
zdist.DeltaT

# The sky is >~ 40000 sq degrees ~ 4000 * LSST field of view
zdist.skyFraction * 2000 * 2

zdist.zbinEdges

zdist.zSampleSize().sum()

zdist.zbinEdges

fig, ax = plt.subplots(1, 2)
_ = ax[0].hist(zdist.zSamples, bins=np.arange(0., 1.1, 0.1), histtype='step', lw=2)
_ = ax[0].errorbar(0.5*(zdist.zbinEdges[:-1]+zdist.zbinEdges[1:]), zdist.zSampleSize(),
                   yerr=np.sqrt(zdist.zSampleSize()), fmt='o')
_ = ax[1].plot(0.5*(zdist.zbinEdges[:-1]+zdist.zbinEdges[1:]), 
               18000*zdist.zSampleSize()/ zdist.fieldArea, 'ko')

