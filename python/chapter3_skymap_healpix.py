# Basic notebook imports
get_ipython().magic('matplotlib inline')

import matplotlib
import pylab as plt
import numpy as np
import healpy as hp

import skymap
from skymap import Skymap,McBrydeSkymap,OrthoSkymap
from skymap import SurveySkymap,SurveyMcBryde,SurveyOrtho
from skymap import DESSkymap,BlissSkymap

SKYMAPS = [Skymap,McBrydeSkymap,OrthoSkymap]
SURVEYS = [SurveySkymap,SurveyMcBryde,SurveyOrtho]

NSIDE = 8

hpxmap = np.arange(hp.nside2npix(NSIDE))
fig,axes = plt.subplots(1,3,figsize=(20,4))
for i,cls in enumerate(SKYMAPS):
    plt.sca(axes[i])
    m = cls()
    m.draw_hpxmap(hpxmap,xsize=200)
    plt.title('HEALPix Map (%s)'%cls.__name__)

hpxmap = np.arange(hp.nside2npix(NSIDE))
fig,axes = plt.subplots(1,3,figsize=(20,4))
for i,proj in enumerate(['sinu','kav7','vandg']):
    plt.sca(axes[i])
    smap = Skymap(projection=proj,lon_0=0)
    im,lon,lat,values = smap.draw_hpxmap(hpxmap,xsize=200)
    plt.title('HEALPix Map (%s)'%proj)

plt.figure()
plt.pcolormesh(lon,lat,values)

pix = hpxmap = np.arange(525,535)
fig,axes = plt.subplots(1,3,figsize=(20,4))
for i,cls in enumerate(SKYMAPS):
    plt.sca(axes[i])
    m = cls()
    m.draw_hpxmap(hpxmap,pix,NSIDE,xsize=200)
    plt.title('Partial HEALPix Map (%s)'%cls.__name__)

pix = np.arange(525,535)

fig,axes = plt.subplots(1,3,figsize=(20,4))
plt.sca(axes[0])
smap = McBrydeSkymap()
hpxmap = hp.UNSEEN * np.ones(hp.nside2npix(NSIDE))
hpxmap[pix] = pix
smap.draw_hpxmap(hpxmap,xsize=400)

plt.sca(axes[1])
smap = McBrydeSkymap()
hpxmap = np.nan * np.ones(hp.nside2npix(NSIDE))
hpxmap[pix] = pix
smap.draw_hpxmap(hpxmap,xsize=400)

plt.sca(axes[2])
smap = McBrydeSkymap()
hpxmap = np.arange(hp.nside2npix(NSIDE))
hpxmap = np.ma.array(hpxmap, mask=~np.in1d(hpxmap,pix))
out = smap.draw_hpxmap(hpxmap,xsize=200)

# These are random, non-uniform points
size = int(1e5)
lon = np.random.uniform(0,360,size=size)
lat = np.random.uniform(-90,90,size=size)

fig,axes = plt.subplots(1,3,figsize=(20,4))
for i,cls in enumerate(SKYMAPS):
    plt.sca(axes[i])
    smap = cls()                                                  
    hpxmap,(im,lon,lat,values) = smap.draw_hpxbin(lon,lat,nside=16,xsize=200)
    plt.title('HEALPix Binning (%s)'%cls.__name__)

import skymap.healpix

ra,dec = 45,-45
fig,axes = plt.subplots(1,3,figsize=(15,4))

for i,nside in enumerate([512,4096,4096*2**5]):
    radius = np.degrees(50*hp.nside2resol(nside))
    pixels = skymap.healpix.ang2disc(nside,ra,dec,radius)
    values = pixels
    
    plt.sca(axes[i])
    # Use the Cassini projection (because we can)                                
    m = Skymap(projection='cass', lon_0=ra, lat_0=dec, celestial=False,
               llcrnrlon=ra+2*radius,urcrnrlon=ra-1.5*radius,
               llcrnrlat=dec-1.2*radius,urcrnrlat=dec+1.5*radius,
               parallels=False, meridians=False)

    m.draw_hpxmap(values,pixels,nside=nside,xsize=200)
    m.draw_parallels(np.linspace(dec-2*radius,dec+2*radius,5),
                     labelstyle='+/-',labels=[1,0,0,0],fmt = '%.2f')
    m.draw_meridians(np.linspace(ra-2*radius,ra+2*radius,5),
                     labelstyle='+/-',labels=[0,0,0,1],fmt = '%.2f')

    plt.title('HEALPix Zoom (nside=%i)'%nside)

