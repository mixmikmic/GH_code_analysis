import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from itertools import repeat
import pyoorb as oo

from moObs import MoOrbits, MoObs

orbitfile = 'pha20141031.des'
mo = MoObs()
mo.readOrbits(orbitfile)

print mo.orbits.columns.values

# Set up to run OpenOrb to generate ephemerides.
mo.setupOorb()

timestep = 2.0/24.0
mo.setTimes(timestep=timestep, ndays=10, timestart=49353.16)

sso = mo.orbits.iloc[0]
oorbelems = mo._packOorbElem(sso)

oorbephs = mo._generateOorbEphs(oorbelems)
ephs = mo._unpackOorbEphs(oorbephs)

# Currently linear interpolation
interpfuncs = mo.interpolateEphs(ephs)

orbsubset = mo.orbits[0:10]
ephTimes = mo.setTimes(timestep=timestep, ndays=365)
print len(mo.ephTimes.swapaxes(1, 0)[0])

get_ipython().run_cell_magic('timeit', '', 'ephs = mo.generateEphs(orbsubset)')

get_ipython().run_cell_magic('timeit', '', 'for i, sso in orbsubset.iterrows():\n    ephs = mo.generateEphs(sso)')

a = np.array([2.4, 3.95, 7.9, 15.7])
b = np.array([2.55, 4.15, 8.37, 16.6])
x = np.array([30, 50, 100, 200])
plt.plot(x, a/x)
plt.plot(x, b/x)
print (b/x - a/x), a/x
print 'Per object, grouping objects rather than looping is', np.mean((b/x-a/x) / (a/x) * 100), '% faster.'
print ' (difference is about ', np.mean(b/x-a/x), 's faster, out of', np.mean(a/x), 's total time'

# Set up for an accuracy test.
#orbitfile = 'pha20141031.des'
#orbitfile = 'mbas_1000.txt'
orbitfile = 'tnos_1000.txt'

mo = MoObs()
mo.readOrbits(orbitfile)
timestart = mo.orbits['epoch'].min()
mo.setTimes(timestep=.2/24.0, ndays=0.02*365, timestart=timestart + 5.0)
testEphTimes = np.copy(mo.ephTimes)
testtimes = testEphTimes.swapaxes(1, 0)[0]
mo.setTimes(timestep=24*2/24.0, ndays=0.02*365+10., timestart=timestart)
ephTimes = np.copy(mo.ephTimes)
print ephTimes.shape, testEphTimes.shape

print 'Test time range', testtimes.min(), testtimes.max()
print 'Time range for interpolation', ephTimes.swapaxes(1, 0)[0].min(), ephTimes.swapaxes(1, 0)[0].max()
print 'Number of orbits to test', len(mo.orbits)

diff = {}
for col in ephs.dtype.names:
    if col == 'time':
        continue
    diff[col] = np.zeros(len(mo.orbits))
for i, orb in mo.orbits.iterrows():
    mo.ephTimes = ephTimes
    ephs = mo.generateEphs(orb)
    interpfuncs = mo.interpolateEphs(ephs)
    # Evaluate errors
    mo.ephTimes = testEphTimes
    testephs = mo.generateEphs(orb)
    for col in testephs.dtype.names:
        if col == 'time':
            continue
        val = interpfuncs[col](testtimes)
        diff[col][i] = np.max(np.abs(val - testephs[col][0]))

print diff['ra'].shape
# Verify small errors
for col in diff:
    print col, diff[col].max(), diff[col].max()*60.*60., np.median(diff[col]), np.median(diff[col]*60.*60.)

for col in ephs.dtype.names:
    if col == 'time':
        continue
    plt.figure()
    plt.plot(ephs['time'][0], ephs[col][0], 'r.')
    if col in interpfuncs:
        plt.plot(ephs['time'][0], interpfuncs[col](ephs['time'][0]), 'b-')
    plt.xlabel('time')
    plt.ylabel(col)
    #plt.xlim(51230, 51280)

plt.plot(ephs['ra'][0], ephs['dec'][0], 'r.')

print interpfuncs.keys()

# Test Ceres against Horizons predictions (in particular solar elongation)
ceres = MoObs()
ceres.readOrbits('ceres.des')
print ceres.orbits['g']

ceres.setTimes(timestep=1, timestart=57209.0, ndays=365*10)
ceres_ephs = ceres.generateEphs()

import pandas as pd
ceph = pd.read_csv('horizons_ceres.txt', skipinitialspace=True)

ceph.head(3)

plt.plot(ceres_ephs[0]['time'], ceres_ephs[0]['solarelon'], 'k.')
plt.plot(ceph['Date_________JDUT']-2400000.5, ceph['S-O-T'], 'r.')



