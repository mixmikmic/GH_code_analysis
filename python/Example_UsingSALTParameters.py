import numpy as np

import snsims
import healpy as hp

from astropy.cosmology import Planck15 as cosmo

import sncosmo

zdist = snsims.PowerLawRates(rng=np.random.RandomState(1), 
                             fieldArea=9.6,
                             surveyDuration=10.,
                             zbinEdges=np.arange(0.10001, 1.1, 0.1))

zdist.zSamples

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

sp = snsims.SimpleSALTDist(numSN=len(zdist.zSamples), zSamples=zdist.zSamples, rng=np.random.RandomState(8))

sp.paramSamples.head()

mB = np.zeros(len(sp.paramSamples))

model = sncosmo.Model('salt2')
for ind, row in sp.paramSamples.iterrows():
    model.set(**dict(z=row['z'], x1=row['x1'], c=row['c'], x0=row['x0']))
    mB[ind] = model.source_peakabsmag('bessellB', 'ab')

sp.paramSamples['mBessellB'] = mB

gmm_dist = snsims.GMM_SALT2Params(numSN=len(zdist.zSamples), 
                                  zSamples=zdist.zSamples, 
                                  rng=np.random.RandomState(8))

gp = gmm_dist.paramSamples

gmm_dist.zSamples

sp.paramSamples.mBessellB.describe()

gp.mB.describe()

fig, ax = plt.subplots(1, 3)
sns.distplot(sp.paramSamples.mBessellB, rug=False, hist_kws=dict(histtype='step', alpha=1, lw=0.5), color='r', ax=ax[0])
sns.distplot(gp.mB, rug=False, hist_kws=dict(histtype='step', alpha=1, lw=0.5), color='k', ax=ax[0], label='')

sns.distplot(sp.paramSamples.x1, rug=False, hist_kws=dict(histtype='step', alpha=1, lw=0.5), color='r', ax=ax[1])
sns.distplot(gp.x1, rug=False, hist_kws=dict(histtype='step', alpha=1, lw=0.5), color='k', ax=ax[1])

sns.distplot(sp.paramSamples.c, rug=False, hist_kws=dict(histtype='step', alpha=1, lw=0.5), color='r')
sns.distplot(gp.c, rug=False, hist_kws=dict(histtype='step', alpha=1, lw=0.5), color='k')
plt.legend()

sns.pairplot(gp[['mB', 'x1', 'c', 'z']], kind='scatter')

sns.pairplot(sp.paramSamples[['mBessellB', 'x1', 'c', 'z']], kind='scatter')

import sncosmo
model = sncosmo.Model(source='salt2')

model.set(z=0.5, x1=0., c=0.)

model.set_source_peakabsmag(-19.0, 'bessellB', 'ab')

print model

model.source.peakmag('bessellB', 'ab')

from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=69.3, Om0=0.286)

cosmo.distmod(z=0.5).value -19.0



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



