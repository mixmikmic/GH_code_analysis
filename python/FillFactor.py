import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import itertools

# For the footprint generation and conversion between galactic/equatorial coordinates.
from lsst.sims.utils import haversine, ObservationMetaData
from lsst.obs.lsstSim import LsstSimMapper
from lsst.sims.coordUtils import _observedFromICRS, _chipNameFromRaDec

import lsst.sims.utils as sims_utils

mapper = LsstSimMapper()
camera = mapper.camera
epoch = 2000.0
site = sims_utils.Site()

# Set up the observation metadata (boresight + time + rotator angle.)
fieldra = np.radians(10.0)
fielddec = np.radians(-0.0)
expmjd = 52999.96
alt, az, pa = sims_utils._altAzPaFromRaDec(fieldra, fielddec, site.longitude, site.latitude, expmjd)
print np.degrees(alt)

rotskypos = np.radians(0.0)
obs_metadata = ObservationMetaData(unrefractedRA = np.degrees(fieldra),
                                   unrefractedDec = np.degrees(fielddec),
                                   rotSkyPos = np.degrees(rotskypos),
                                   mjd = expmjd)

# Set up points to iterate over (to calculate fill factor)
testrange = 2.2
spacing = 0.01
decscale = np.cos(fielddec)
rai = np.arange(fieldra - np.radians(testrange)*decscale, 
                fieldra + np.radians(testrange)*decscale, np.radians(spacing)*decscale)
deci = np.arange(fielddec - np.radians(testrange), 
                 fielddec + np.radians(testrange), np.radians(spacing))
ra = []
dec = []
for i in itertools.product(rai, deci):
    ra.append(i[0])
    dec.append(i[1])
ra = np.array(ra)
dec = np.array(dec)

# Calculate 'visibility'
refractedra, refracteddec = _observedFromICRS(ra, dec, obs_metadata=obs_metadata, epoch=epoch)
chipNames = _chipNameFromRaDec(ra=refractedra, dec=refracteddec, epoch=epoch, 
                                camera=camera, obs_metadata=obs_metadata)
vis = np.zeros(len(chipNames))
for i, chip in enumerate(chipNames):
    if chip is not None:
        vis[i] = 1

# Calculate % fill factor, for a particular assumed (circular) fov
area_degsq = 9.6
radius = np.sqrt(area_degsq/np.pi)
print 'radius matching area_degsq for circle', radius
sep = np.degrees(sims_utils.haversine(ra, dec, fieldra, fielddec))
print 'max distance from center used for chip calculation', sep.max()
incircle = np.where(sep < radius)[0]
onchip = np.where(vis[incircle] == 1)[0]
fillfactor = len(vis[onchip]) / float(len(vis[incircle]))
print 'fill factor', fillfactor

# Plot results.
plt.figure()
plt.axis('equal')
condition = np.where(vis == 1)[0]
# Use 'offset' so that plot doesn't cross 0
offset = 0
plt.plot((np.degrees(ra)+offset)%360, np.degrees(dec), 'g.', markersize=0.2)
plt.plot((np.degrees(ra[condition])+offset)%360, np.degrees(dec[condition]), 'b.', markersize=0.3)
plt.plot(np.degrees(fieldra)+offset, np.degrees(fielddec), 'r.')
plt.xlabel('RA')
plt.ylabel('Dec')
theta = np.arange(0, 2*np.pi, 0.01)
plt.plot(radius*np.cos(theta)+np.degrees(fieldra)+offset, radius*np.sin(theta)+np.degrees(fielddec), 'r-')
outerradius = 2.06
plt.plot(outerradius*np.cos(theta)+np.degrees(fieldra)+offset, outerradius*np.sin(theta)+np.degrees(fielddec), 'r:')
#plt.xlim(fieldra+offset-sep.max()/2.0, fieldra+offset+sep.max()/2.0)
plt.ylim(np.degrees(fielddec)-sep.max()*.75, np.degrees(fielddec)+sep.max()*.75)

# zoom in to see a raft (ccd = 13', raft = 13*3 ' = 0.6 deg) in higher detail.
testrange = 0.5
spacing = 0.001
decscale = np.cos(fielddec)
rai = np.arange(fieldra - np.radians(testrange)*decscale, 
                fieldra + np.radians(testrange)*decscale, np.radians(spacing)*decscale)
deci = np.arange(fielddec - np.radians(testrange), 
                 fielddec + np.radians(testrange), np.radians(spacing))
ra = []
dec = []
for i in itertools.product(rai, deci):
    ra.append(i[0])
    dec.append(i[1])
ra = np.array(ra)
dec = np.array(dec)

# Calculate 'visibility'
refractedra, refracteddec = _observedFromICRS(ra, dec, obs_metadata=obs_metadata, epoch=epoch)
chipNames = _chipNameFromRaDec(ra=refractedra, dec=refracteddec, epoch=epoch, camera=camera, obs_metadata=obs_metadata)
vis = np.zeros(len(chipNames))
for i, chip in enumerate(chipNames):
    if chip is not None:
        vis[i] = 1

plt.figure()
plt.axis('equal')
condition = np.where(vis == 1)[0]
# Use 'offset' so that plot doesn't cross 0
offset = 0
plt.plot((np.degrees(ra)+offset)%360, np.degrees(dec), 'g.', markersize=0.2)
plt.plot((np.degrees(ra[condition])+offset)%360, np.degrees(dec[condition]), 'b.', markersize=0.3)
plt.plot(np.degrees(fieldra)+offset, np.degrees(fielddec), 'r.')
plt.xlabel('RA')
plt.ylabel('Dec')
plt.xlim(9.5, 10.5)
plt.ylim(-.5, 0.5)

# And zoom in even more, to check raft/ccd spacing. Do single line in RA.
testrange = 2.0
spacing = 0.00005
decscale = np.cos(fielddec)
ra = np.arange(fieldra - np.radians(testrange)*decscale, 
                fieldra + np.radians(testrange)*decscale, np.radians(spacing)*decscale)
dec = np.zeros(len(ra)) + fielddec

# Calculate 'visibility'
refractedra, refracteddec = _observedFromICRS(ra, dec, obs_metadata=obs_metadata, epoch=epoch)
chipNames = _chipNameFromRaDec(ra=refractedra, dec=refracteddec, epoch=epoch, camera=camera, obs_metadata=obs_metadata)
vis = np.zeros(len(chipNames))
for i, chip in enumerate(chipNames):
    if chip is not None:
        vis[i] = 1

plt.plot(ra, vis)
condition = np.where(np.diff(vis) != 0)[0]
print np.diff(np.degrees(ra[condition])*60.*60.)

# I need to map the overall visibility back into a 100x100 grid for Peter.



