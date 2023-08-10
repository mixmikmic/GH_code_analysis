get_ipython().magic('matplotlib inline')

import phoebe
from phoebe import u # units
import numpy as np
import matplotlib.pyplot as plt

logger = phoebe.logger()

b = phoebe.default_binary()

b.run_compute(protomesh=True)

print b['model'].kinds

print b['model'].datasets

b.filter(dataset='protomesh', context='model')

b.filter(dataset='protomesh', context='model', component='primary')

b.get_value(dataset='protomesh', context='model', component='primary', qualifier='teffs')

axs, artists = b.filter(dataset='protomesh', context='model', component='secondary').plot(facecolor='teffs', edgecolor=None)

b.add_dataset('lc', times=[0,1,2], dataset='lc01')

b.run_compute(pbmesh=True)

print b['model'].kinds

print b['model'].datasets

b.filter(dataset='pbmesh', context='model')

b.filter(dataset='pbmesh', context='model', component='primary')

b.filter(kind='mesh', context='model', component='primary')

b.filter(dataset='lc01', kind='mesh', context='model', component='primary')

axs, artists = b.filter(kind='mesh', context='model', time=1.0).plot(facecolor='intensities@lc01', edgecolor='teffs')

b.get_value('times@lc01@dataset')

b.add_dataset('mesh', times=[0.5, 1.5], dataset='mesh01')

b.run_compute(protomesh=False, pbmesh=False)

print b['model'].kinds

print b['model'].datasets

b.filter(kind='mesh', context='model').times

b.get_value(kind='mesh', context='model', dataset='lc01', time=0.5, qualifier='intensities', component='primary')

b.filter(dataset='mesh01', kind='mesh', context='model', component='primary', time=0.5)

axs, artists = b.filter(dataset='mesh01', kind='mesh', context='model', time=0.5).plot(facecolor='teffs', edgecolor=None)

b.run_compute(pbmesh=True)

b.filter(kind='mesh', context='model').times



