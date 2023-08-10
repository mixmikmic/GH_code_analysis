import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from moObs import MoObs

#orbitfile = 'GreenstreetPHAs.des'
#orbitfile = 'pha20141031.des'
orbitfile = 'test.des'
moo = MoObs()
moo.readOrbits(orbitfile)

# See what we've got in the orbit data.
print moo.orbits.columns.values
moo.orbits.head()

# timesteps of 2 hours for NEOS; 5 hours for MBAS and 2 days for TNOs seems pretty good
moo.setTimes(timestep = 2./24., ndays=365*10, timestart = 49353.0)
moo.setupOorb()

get_ipython().run_cell_magic('timeit', '', 'for i, sso in moo.orbits.head(3).iterrows():\n    ephs = moo.generateEphs(sso)\n    interpfuncs = moo.interpolateEphs(ephs)')

from lsst.sims.maf.db import OpsimDatabase

opsdb = OpsimDatabase('/Users/lynnej/opsim/db/enigma_1189_sqlite.db')

# What columns from opsim do you want to have available when doing later 'metric' generation?
dbcols = ['expMJD', 'night', 'fieldRA', 'fieldDec', 'rotSkyPos', 'filter',
          'finSeeing', 'fiveSigmaDepth', 'visitExpTime', 'solarElong']
simdata = opsdb.fetchMetricData(dbcols, sqlconstraint='')

#%%timeit
outfileName = 'test_out.txt'
try:
    del moo.outfile
except AttributeError:
    pass
useCamera=True
# Limit this example to only calculating positions for the first few objects.
for i, sso in moo.orbits.iterrows():
    ephs = moo.generateEphs(sso)
    interpfuncs = moo.interpolateEphs(ephs)
    # This is how you change the matching against the FOV. 
    #  If useCamera=True, then rFov is irrelevant - the silicon footprint is used.
    #  If useCamera=False, then rFov is used instead, as a circular footprint. 
    idxObs = moo.ssoInFov(interpfuncs, simdata, rFov=np.radians(1.75), useCamera=useCamera)
    moo.writeObs(sso['objId'], interpfuncs, simdata, idxObs, outfileName=outfileName)

# See what this output file looks like. 
# This file stores the moving object observations - even if the object would have been too faint!
get_ipython().system('head test_out.txt')



