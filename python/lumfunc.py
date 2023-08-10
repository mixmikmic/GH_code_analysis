import os

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('pylab inline')

from astropy.io.ascii import read
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
import fitsio
import seaborn

# Reading in the RedMaPPer iSEDfit catalog.
rmpath = os.path.join(os.sep, 'global', 'work', 'projects', 'redmapper')
rmcatfile = os.path.join(rmpath, 'redmapper_isedfit_v5.10_centrals.fits.gz')
rminfo = fitsio.FITS(rmcatfile)
rmcat = rminfo[1].read(columns=['Z', 'MSTAR_50', 'ILUM', 'LAMBDA_CHISQ', 'P_CEN', 'P_SAT', 'MSTAR_AVG', 'P_BCG', 'P_C'])

# WARNING! This cell is huge. It takes a long time to run.
#satellites = os.path.join(rmpath, 'redmapper_isedfit_v5.10_satellites.fits.gz')
#satinfo = fits.open(satellites)

#satinfo[1].header

def indhist(data, bounds):
    catalog = data
    indices = np.where((catalog >= bounds[0]) * (catalog <= bounds[1]))[0]
    return indices

def getdata(physics):
    for jj in range(numslices):
        physics[jj+1] = indhist(rmcat['ILUM'][keep], [masschunks[1][jj], masschunks[1][jj+1]])
    return;

# Convert dictionary to matrix
def dicttomatrix(dictionary):
    dataarray = []
    for ii in range(numslices):
        dataarray.append(np.array(list(dictionary.items())[ii][1]))
    return dataarray

# This function accepts two 1x2 arrays. The first is the richness bounds. The second is the redshift bounds.
def cutter(lbounds, zbounds):
    keep = np.where((rmcat['LAMBDA_CHISQ'] > lbounds[0]) * (rmcat['LAMBDA_CHISQ'] < lbounds[1]) * 
                    (rmcat['Z'] > zbounds[0]) * (rmcat['Z'] < zbounds[1]) * (rmcat['MSTAR_50'] > 10)
                   * (rmcat['MSTAR_50'] < 13))[0]
    return keep

# Goal: Create some clean subplots, reminiscent of those in the paper (but better).
# How do I compute the Conditional Mass Function (CMF)?
# Create little slices of richness and redshift.
# Figure out which sources are in each slice.
# Compute the CMF of each slice.
# Plot CMF vs. M*.

masses = {}
numslices = 100

keep = cutter([0, 100], [0.1, 0.3])
fig, ax = plt.subplots()
ax.scatter(rmcat['Z'][keep], rmcat['LAMBDA_CHISQ'][keep])

masssep = np.linspace(min(rmcat['ILUM'][keep]), max(rmcat['ILUM'][keep]), numslices)
mdiff = np.log10(masssep[1])-np.log10(masssep[0])
masschunks = plt.hist(rmcat['ILUM'][keep], bins=numslices)

getdata(masses)
massarray = dicttomatrix(masses)

centralcmf = []
satcmf = []

plt.figure()
for ii in range(numslices):
    if len(massarray[ii] > 0):
        phic = np.sum(rmcat['P_CEN'][massarray[ii]])/(len(rmcat['ILUM'][keep])*mdiff)
        phis = np.sum(rmcat['P_SAT'][massarray[ii]])/(len(rmcat['ILUM'][keep])*mdiff)
        centralcmf.append(phic)
        satcmf.append(phis)
        phi = phic + phis 
        plt.plot(masschunks[1][ii], phi, 'bo', label=r'$\Phi$')
        #plt.plot(masschunks[1][ii], phic, 'b^', label=r'$\Phi_C$')
        #plt.plot(masschunks[1][ii], phis, 'rs', label=r'$\Phi_S$')
    if ii == 0:
        plt.legend(loc=1)
plt.xlabel(r'$\log \tt{L}^*$')
plt.ylabel(r'$\Phi$')



