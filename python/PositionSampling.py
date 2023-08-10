import snsims

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

import healpy as hp

import numpy as np

NSIDE=4

theta_c, phi_c = hp.pix2ang(nside=NSIDE, ipix=1, nest=True)

print(theta_c, phi_c)

rad = np.sqrt(hp.nside2pixarea(NSIDE, degrees=True)/ np.pi)

rng = np.random.RandomState(1)

phi_, theta_ = snsims.Tiling.samplePatchOnSphere(np.degrees(phi_c), np.degrees(theta_c), delta=rad, size=1000, 
                                                 rng=rng, degrees=False)

np.unique(hp.ang2pix(NSIDE,theta_, phi_, nest=True))

import opsimsummary as oss

import os

datadir = os.path.join(oss.__path__[0], 'example_data')
opsimdb = os.path.join(datadir, 'enigma_1189_micro.db')

hpOpSim = oss.HealPixelizedOpSim.fromOpSimDB(opsimdb, NSIDE=NSIDE)

hpTiles = snsims.HealpixTiles(nside=NSIDE, healpixelizedOpSim=hpOpSim)

Phi_, Theta_ = hpTiles._angularSamples(np.degrees(phi_c), 
                                       np.degrees(theta_c), 
                                       radius=rad,
                                       numSamples=100000,
                                       tileID=1,
                                       rng=np.random.RandomState(1))

len(Phi_)

np.unique(hp.ang2pix(NSIDE, Theta_, Phi_, nest=True))

p, t = hpTiles.positions(1, 100000, rng=np.random.RandomState(1))

len(p)

np.radians(p[:len(Phi_)]) == Phi_

np.unique(hp.ang2pix(NSIDE, np.radians(t), np.radians(p), nest=True))

fig, ax = plt.subplots()
#_ = ax.scatter(theta_, phi_)
#_ = ax.plot(Theta_, Phi_, 'rs')
_ = ax.plot(np.radians(t), np.radians(p), '.g')

x= np.arange(0., 1.4, 0.1)

mapvalues = np.ones(hp.nside2npix(NSIDE)) * hp.UNSEEN
mapvalues[1] = 1.

hp.mollview(mapvalues, nest=True)
#hp.projscatter(theta_c, phi_c)
# hp.projscatter(Theta_, Phi_)
hp.projscatter(np.radians(t), np.radians(p))

hp.mollview(mapvalues, nest=True)

fig, ax = plt.subplots()
_ = ax.hist(np.degrees(np.arccos(np.dot(hp.ang2vec(theta_, phi_), hp.ang2vec(theta_c, phi_c)))), 
            bins=10, histtype='step', normed=1)



