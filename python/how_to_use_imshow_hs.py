# import numpy and matplotlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
# Import local module
import graphics

# Check matplotlib version
matplotlib.__version__

# load magnetic data (USGS survey data)
mag = np.load(r'.\data\mag4017.npy')

# Note: the colormap is set to 'jet' to simulate the behaviour of pyplot prior to version 2.0.0
plt.imshow(mag,cmap='jet')

plt.imshow(mag,cmap='viridis',extent=(-308000.0, -183500.0, 0.0, 61500.0),origin='lower')

fig,ax = plt.subplots(figsize=(12,6))
graphics.imshow_hs(mag,ax,cmap='viridis',cmap_norm='nonorm',hs=False,colorbar=True,
                   extent=(-308000.0, -183500.0, 0.0, 61500.0),origin='lower')

fig,ax = plt.subplots(figsize=(12,6))
graphics.imshow_hs(mag,ax,cmap='viridis',cmap_norm='nonorm',hs=False,colorbar=True,
                   cb_ticks='stats',nSigma=2,
                   extent=(-308000.0, -183500.0, 0.0, 61500.0),origin='lower')

fig,ax = plt.subplots(figsize=(12,6))
graphics.imshow_hs(mag,ax,cmap='viridis',cmap_norm='equalize',hs=False,colorbar=True,
                   cb_ticks='stats',nSigma=2,
                   extent=(-308000.0, -183500.0, 0.0, 61500.0),origin='lower')

fig,ax = plt.subplots(figsize=(12,6))
graphics.imshow_hs(mag,ax,cmap='geosoft',cmap_norm='equalize',hs=False,colorbar=True,
                   cb_ticks='stats',nSigma=2,
                   extent=(-308000.0, -183500.0, 0.0, 61500.0),origin='lower')

fig,ax = plt.subplots(figsize=(12,6))
graphics.imshow_hs(mag,ax,cmap='viridis',cmap_norm='autolevels',hs=False,colorbar=True,
                   cb_ticks='stats',nSigma=2,minPercent=2,maxPercent=98,
                   extent=(-308000.0, -183500.0, 0.0, 61500.0),origin='lower')

fig,ax = plt.subplots(figsize=(12,6))
graphics.imshow_hs(mag,ax,cmap='coolwarm',cmap_norm='autolevels',hs=False,colorbar=True,
                   cb_ticks='stats',nSigma=2,minPercent=2,maxPercent=98,
                   contours=True,levels=[-600,-300,0,300,600],
                   extent=(-308000.0, -183500.0, 0.0, 61500.0),origin='lower')

fig,ax = plt.subplots(figsize=(12,6))
graphics.imshow_hs(mag,ax,cmap='coolwarm',cmap_norm='autolevels',hs=True,colorbar=True,
                   cb_ticks='stats',nSigma=2,azdeg=45,altdeg=45,blend_mode='soft',
                   extent=(-308000.0, -183500.0, 0.0, 61500.0),origin='lower')

fig,ax = plt.subplots(figsize=(12,6))
graphics.imshow_hs(mag,ax,cmap='geosoft',cmap_norm='equalize',hs=True,colorbar=True,
                   cb_ticks='stats',nSigma=2,azdeg=45,altdeg=45,blend_mode='alpha',alpha=0.7,
                   extent=(-308000.0, -183500.0, 0.0, 61500.0),origin='lower')

# first create a new normalization using discrete boundaries
import matplotlib.colors as mcolors
mynorm = mcolors.BoundaryNorm(graphics.stats_boundaries(mag,nSigma=2,sigmaStep=0.5),256)

fig,ax = plt.subplots(figsize=(12,6))
graphics.imshow_hs(mag,ax,cmap='coolwarm',cmap_norm='no',hs=True,colorbar=True,norm=mynorm,
                   cb_ticks='stats',nSigma=2,azdeg=45,altdeg=45,blend_mode='alpha',alpha=0.7,
                   extent=(-308000.0, -183500.0, 0.0, 61500.0),origin='lower')



