import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import os
from astropy.io import fits

from astropy.table import Table
import healpy as hp
from collections import OrderedDict
from matplotlib import cm
import copy

from dataCleanUp import dataCleanUp
from addSNR import addSNR
from createMaps import createMeanStdMaps
from mpl_toolkits.axes_grid1 import ImageGrid
from twoPtCorr import runTreeCorr
from plots import plot_wtheta
import flatmaps as fm
 
import intermediates

SNRthreshold= 10

HSCdatapath= '/global/cscratch1/sd/damonge/HSC/'
HSCFiles= os.listdir(HSCdatapath)
HSCFiles= ['HSC_WIDE_GAMA15H_forced.fits', 'HSC_WIDE_GAMA15H_random.fits']

HSCFiles= [HSCdatapath+f for f in HSCFiles]
HSCFiles

HSCdata= {}
for filename in HSCFiles:
    key= filename.split('WIDE_')[1].split('.fits')[0]
    print 'Reading ', filename
    dat = Table.read(filename, format='fits')
    HSCdata[key] = dat.to_pandas()
    
HSCFieldTag= key.split('_')[0]  # just the field tag.

# clean up
for key in HSCdata:
    print key
    HSCdata[key]= dataCleanUp(HSCdata[key])

outputDir= 'flatmaps_nside1024/'
depthFiles= os.listdir(outputDir)
depthFiles= [f for f in depthFiles if not f.__contains__('std') and 
             f.__contains__('%ssigma'%SNRthreshold)] # ignore the std across the maps

depthMap= {}

for filename in depthFiles:
    print 'reading ', filename
    splits= filename.split('-band_')
    band= splits[0][-1:]
    method= splits[1].split('Method')[0]
    
    if method not in depthMap.keys(): depthMap[method]= {}   
    depthMap[method][band]= fm.read_flat_map(outputDir+filename)[1]

bands= ['i'] #['g', 'r', 'i', 'z', 'y']]
flatSkyGrid= fm.FlatMapInfo([212.5,222.],[-2.,2.], dx=0.057,dy=0.057)

# just plot to see everything is ok
nCols= len(depthMap.keys())
cmap = cm.magma
xlabel, ylabel= 'ra', 'dec'
for band in bands:
    # plot the depth for each method.
    fig = plt.figure(figsize=(15, 10))
    # set up for colorbar
    grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                     nrows_ncols=(1, nCols), axes_pad=0.15, share_all= True, cbar_location="right",
                     cbar_mode="single", cbar_size="5%", cbar_pad=0.15,
                     )
    # since have three maps, need colorange that ~works for all.
    colorMin= 100
    colorMax= -100
    for mInd, method in enumerate(depthMap):
        ind= np.where(depthMap[method][band]>0)[0]
        colorMin= min(colorMin, np.percentile(depthMap[method][band][ind], 0))
        colorMax= max(colorMin, np.percentile(depthMap[method][band][ind], 95))
    # plot
    for mInd, method in enumerate(depthMap):
        ax= grid[mInd]
        image= ax.imshow(depthMap[method][band].reshape([flatSkyGrid.ny,flatSkyGrid.nx]),
                         origin='lower', interpolation='nearest',
                         aspect='equal', extent=[flatSkyGrid.x0, flatSkyGrid.xf, flatSkyGrid.y0, flatSkyGrid.yf],
                         vmin= colorMin, vmax= colorMax, cmap= cmap)
        ax.set_title(method)
        ax.set_xlabel(xlabel,fontsize=15)
        ax.set_ylabel(ylabel,fontsize=15)
        ax.cax.colorbar(image)
        ax.cax.toggle_label(True)
    plt.suptitle('%s-band %sigma depth'%(band, SNRthreshold), fontweight="bold", )
    plt.subplots_adjust(top=1.6)
    plt.show()
   

HSCFieldTag= 'GAMA15H_forced'
band= 'i'

pixelNums= flatSkyGrid.pos2pix(HSCdata[HSCFieldTag]['ra'], HSCdata[HSCFieldTag]['dec'])

# find five sigma depth for each object. add columns to dataframe.
fiveSigCols= {}
for method in depthMap:
    fiveSigCols[method]= []
    print method
    for band in bands:
        key= '%s-%s-%ssigmaDepth'%(method, band, SNRthreshold)
        fiveSigCols[method].append(key)
        HSCdata[HSCFieldTag][key]= depthMap[method][band][pixelNums]
    
    print 'Added cols: %s'%fiveSigCols[method]

# plot histogram: number of objects we have for different depth bins
fig, ax = plt.subplots()
for method in depthMap:
    for key in fiveSigCols[method]:
        plt.hist(HSCdata[HSCFieldTag][key], label= method,
                 histtype= 'step', alpha= 1., lw= 3, bins= 5)
ax.legend(loc= "upper left")
ax.set_xlabel('%s$\sigma$ Depth'%SNRthreshold)
ax.set_ylabel('Number of objects')
ax.set_title(key.split(method+'-')[1])
plt.show()

from collections import OrderedDict
densityTrend= OrderedDict()
for method in depthMap:
    fiveSigmaKey= '%s-%s-10sigmaDepth'%(method, band)
    densityTrend[method]= intermediates.getDensity_magCuts(depthmap= depthMap[method][band],
                                                           band= band,
                                                           galDepth= HSCdata[HSCFieldTag][fiveSigmaKey],
                                                           galMags= HSCdata[HSCFieldTag]['%scmodel_mag'%band],
                                                           depthRange= [25, 27], deldepthStep= 0.3,
                                                           plotTitle= method)

limitingMag= {'Javis': 25.0, 'FluxErr': 25.0, 'dr1paper': 25.0}

HSCFieldTag= 'GAMA15H_forced'

# find the mag-limited galaxy sample
galSample_ra, galSample_dec= {}, {}

for method in limitingMag:
    print method
    ind= np.where((HSCdata[HSCFieldTag]['%scmodel_mag'%band]<limitingMag[method]) & 
              (HSCdata[HSCFieldTag]['iclassification_extendedness']==1))[0]

    galSample_ra[method], galSample_dec[method]= HSCdata[HSCFieldTag]['ra'].iloc[ind], HSCdata[HSCFieldTag]['dec'].iloc[ind]
    print '%s galaxies to consider.'%len(galSample_ra[method])
    
    plt.clf()
    sns.jointplot(x=galSample_ra[method], y= galSample_dec[method], kind="hex", color="k")
    plt.show()

# random depth maps
masks= {}
for method in depthMap.keys():
    print method
    masks[method]= intermediates.getMask(depthMap[method][band], band, limitingMag[method],
                                         plotMap= True, flatSkyGrid= flatSkyGrid, title= method)

nData= 0
for method in depthMap:
    nData= max(nData, len(galSample_ra[method]))

randCatalog= {}
for method in depthMap:
    randCatalog[method]= intermediates.getRandomCatalog(flatSkyGrid, masks[method],
                                                        minRA= min(HSCdata[HSCFieldTag]['ra']),
                                                        maxRA= max(HSCdata[HSCFieldTag]['ra']),
                                                        minDec=  min(HSCdata[HSCFieldTag]['dec']),
                                                        maxDec=  max(HSCdata[HSCFieldTag]['dec']),
                                                        nData= nData, plotMap= False)

theta, wtheta, wtheta_sig= {}, {}, {}
for method in depthMap:
    theta[method], wtheta[method], wtheta_sig[method]= runTreeCorr(data_ra= galSample_ra[method],
                                                                   data_dec= galSample_dec[method],
                                                                   random_ra= randCatalog[method]['ra'],
                                                                   random_dec= randCatalog[method]['dec'],
                                                                   minSep= 0.001, maxSep= 5, nBins= 25)
    
    plot_wtheta(theta[method], wtheta[method], wtheta_sig[method], title= method)

# overplot
for method in theta:
    plt.plot(theta[method], wtheta[method], 'o-', label= method)
plt.legend()
plt.title('Based on %ssigma depth'%SNRthreshold)
plt.xlabel(r'$\theta$ (Degrees)', fontsize= 14)
plt.ylabel(r'$w(\theta)$', fontsize= 14)
plt.loglog()
plt.show()



