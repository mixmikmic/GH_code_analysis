import sys
get_ipython().system('{sys.executable} -m pip install "astropy" "astroquery" "healpy" "matplotlib" "scipy"')

import healpy as hp # for working with HEALPix files
import numpy as np # needed for vector operations
from matplotlib import pyplot as plt # plotting skymaps
from scipy.stats import norm # probability functions

from astropy.utils.data import download_file
url = ('https://dcc.ligo.org/public/0146/G1701985/001/LALInference_v2.fits.gz')
# This is the publication LALInference localization
filename = download_file(url, cache=True)

prob, header = hp.read_map(filename, h=True) # reading in the first column which is the probability skymap and the header

distmu, distsigma, distnorm = hp.read_map(filename, field=[1,2,3])

header

npix = len(prob)
nside = hp.npix2nside(npix)
npix, nside

hp.get_map_size(prob)

hp.get_nside(prob)

maxpix = np.argmax(prob)
maxpix

pixarea = hp.nside2pixarea(nside)
pixarea

pixarea == 4*np.pi/npix

pixarea_deg2 = hp.nside2pixarea(nside, degrees=True)
pixarea_deg2

pixarea_deg2 == pixarea*(180/np.pi)**2

dp_dA = prob[maxpix]/pixarea
dp_dA # Probability per steradian

dp_dA_deg2 = prob[maxpix]/pixarea_deg2
dp_dA_deg2 # Probability per deg^2

ra, dec = 197.45, -23.38 # Coordinates of NGC 4993

# Converting to radians
theta = 0.5*np.pi - np.deg2rad(dec)
phi = np.deg2rad(ra)
theta, phi

ipix = hp.ang2pix(nside, theta, phi)
ipix

dp_dA = prob[ipix]/pixarea # Probability per steradian
dp_dA

dp_dA_deg2 = prob[ipix]/pixarea_deg2 # Probability per deg^2
dp_dA_deg2

r = np.linspace(0,80,100) # Returns 100 evenly spaced numbers between 0 and 80

dp_dr = r**2 * distnorm[ipix] * norm(distmu[ipix], distsigma[ipix]).pdf(r)

plt.plot(r, dp_dr)
plt.xlabel('distance (Mpc)')
plt.ylabel('prob Mpc$^{-1}$')
plt.show()

r = 40

dp_dV = prob[ipix]*distnorm[ipix]*norm(distmu[ipix], distsigma[ipix]).pdf(r)/pixarea
dp_dV

probperdeg2 = prob/pixarea_deg2
hp.mollview(probperdeg2, coord=['C'], title='GW170817 LALInference', max=np.max(probperdeg2))
hp.graticule(local=True)
plt.show()

