get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import os
import cPickle as pickle
from vespa import FPPCalculation, PopulationSet
from vespa.transit_basic import traptransit
from corner import corner

from vespa.populations import ArtificialPopulation, BoxyModel, LongModel

import logging
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.DEBUG)

#koi = 'K05625.01'
#koi = 'K01204.01'
koi = 'K02857.02'
fpp = FPPCalculation.from_ini(koi)
boxmodel = BoxyModel(10./2e5, fpp['pl'].stars.slope.quantile(0.99))
longmodel = LongModel(10./2e5, fpp['pl'].stars.duration.quantile(0.99))

fpp.add_population(boxmodel)
fpp.add_population(longmodel)

fpp2 = FPPCalculation(fpp.trsig, fpp.popset.resample())

c = fpp['eb'].constraints['secondary depth']
c2 = fpp2['eb'].constraints['secondary depth']

c.N

c2.N

c.frac

c2.frac

fpp.prior('heb')

fpp2.prior('heb')

c = fpp['eb'].constraints['secondary depth']

c2 = c.resample([1,2,3,4])

c2.name

c.arrays

rootLogger.setLevel(logging.DEBUG)
fpp.FPP()

def boxy(x, min_slope=10, max_slope=15, logd_range=(-5,0), dur_range=(0,2)):
    """x has shape (3, N), where N is number of shape samples
    
    in 0th dimension, params are duration, logd, slope
    
    flat between min_slope & max_slope, zero below min_slope
    """
    level = 1./((logd_range[1]-logd_range[0])*(dur_range[1]-dur_range[0])*(max_slope-min_slope))
    return level*(x[2,:] > min_slope)

def longdur(x, min_dur=0.5, max_dur=2, slope_range=(2,15), logd_range=(-5,0)):
    """
    flat between min_dur and max_dur, zero below min_dur
    """
    level = 1./((logd_range[1]-logd_range[0])*(slope_range[1]-slope_range[0])*(max_dur-min_dur))
    return level*(x[0,:] > min_dur)

class BoxyModel(ArtificialPopulation):
    max_slope = 15
    logd_range = (-5,0)
    dur_range = (0,2)
    model='boxy'
    modelshort='boxy'

    def __init__(self, prior, min_slope):
        self._prior = prior
        self.min_slope = min_slope
        
    def _lhoodfn(self, x):
        level = 1./((self.logd_range[1]-self.logd_range[0])*
                    (self.dur_range[1]-self.dur_range[0])*
                    (self.max_slope-self.min_slope))
        return level*(x[2,:] > self.min_slope)
        
        
class LongModel(ArtificialPopulation):
    slope_range = (2,15)
    logd_range = (0,5)
    max_dur = 2.
    model='long'
    modelshort='long'
    
    def __init__(self, prior, min_dur):
        self._prior = prior
        self.min_dur = min_dur
        
    def _lhoodfn(self, x):
        level = 1./((self.logd_range[1]-self.logd_range[0])*
                    (self.slope_range[1]-self.slope_range[0])*
                    (self.max_dur-self.min_dur))
        return level*(x[0,:] > self.min_dur)
        

rootLogger.setLevel(logging.INFO)
fpp.FPPsummary()

print BoxyModel(1./len(fpp.popset.poplist), fpp['pl'].stars.slope.quantile(0.99)).lhood(fpp.trsig)
print LongModel(1./len(fpp.popset.poplist), fpp['pl'].stars.duration.quantile(0.99)).lhood(fpp.trsig)
print fpp.lhood('pl')

print fpp['pl'].stars.duration.quantile(0.99)
print fpp['pl'].stars.slope.quantile(0.99)



fpp['pl'].stars.slope.mean()

fpp.lhood('pl')

boxy(fpp.trsig.kde.dataset).sum()

fpp.trsig.kde.dataset.shape

fpp = FPPCalculation.from_ini('K01204.01')

longdur(fpp.trsig.kde.dataset, 0.34).sum()

fpp['pl'].stars.duration.mean()

fpp = FPPCalculation.from_ini('K02857.02')
fig = corner(fpp.trsig.kde.dataset.T, labels=['dur', 'logd', 'slope'], plot_contours=False);
fpp.lhood('pl');

fpp.trsig.MCMC(refit=True, maxslope=30)
corner(fpp.trsig.kde.dataset.T, labels=['dur', 'logd', 'slope'], plot_contours=False);
fpp.lhood('pl')

fpp.lhoodplot('pl')

y = fpp['pl'].kde(fpp.trsig.kde.dataset)
y.sum()/len(y)

def lhood_hist(koi):
    fpp = FPPCalculation.from_ini(koi)

    pl = fpp['pl']
    sig = fpp.trsig
    y = pl.kde(sig.kde.dataset)
    print(y.sum()/len(y))
    pl.lhoodplot(sig)
    plt.figure()
    plt.hist(np.log(y[y>0]));
    corner(sig.kde.dataset.T, labels=['duration', 'log(depth)', 'slope'], plot_contours=False);
    sig.plot(plot_trap=True)
    return y

y0 = lhood_hist('K01204.01')

y1 = lhood_hist('K02857.02')

y2 = lhood_hist('K05625.01')

smin = 2; smax=20
s = np.linspace(smin,smax,100)
plt.plot(s, 1./s)



