import json, urllib, numpy as np, matplotlib.pylab as plt, requests
import pandas as pd
from astropy.io import fits
from matplotlib.cm import register_cmap, cmap_d
from scipy import ndimage
get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format='retina'")

ds9heat = {'red': lambda v : np.interp(v, [0, 0.34, 1], [0, 1, 1]),
           'green': lambda v : np.interp(v, [0, 1], [0, 1]),
           'blue': lambda v : np.interp(v, [0, 0.65, 0.98, 1], [0, 0, 1, 1])}
register_cmap('ds9heat', data=ds9heat)

url = 'http://jsoc.stanford.edu/data/events/Mercury_HMI_I/WholeSun/fits/'
response = urllib.urlopen(url)
times = response.read()
files = pd.read_html(times,skiprows=2)
list_of_files = np.array(files[0][1])
print "There are",len(list_of_files),"files."

background = fits.open("http://jsoc.stanford.edu/data/events/Mercury_HMI_I/WholeSun/fits/20160509_185527_UTC.0554.fits")
background = background[1].data
mask_inverse = np.ones([4096,4096])
masked_image = np.zeros([4096,4096])
composite_dilated_mask = np.zeros([4096,4096])
for i in range(14,len(list_of_files),21):
    if (i == 539): 
        i = 540                                      # capture the egress
    mask = np.zeros([4096,4096])
    continuum = fits.open("http://jsoc.stanford.edu/data/events/Mercury_HMI_I/WholeSun/fits/"+list_of_files[i])
    continuum = continuum[1].data
    differenced_image = background - continuum
    differenced_image[0:1300,:] = 0.0                # mask out the area outside the transit path
    differenced_image[1500:4096,:] = 0.0
    find_mercury = np.where(differenced_image > 14000)
    for i in range(len(find_mercury[0])):
        mask[find_mercury[0][i],find_mercury[1][i]] = 1.
    dilated_mask = ndimage.binary_dilation(mask, iterations=2).astype(mask.dtype)
    composite_dilated_mask = composite_dilated_mask + dilated_mask
    masked_image = continuum*dilated_mask + masked_image

fig = plt.figure(figsize=(14,2.5))
ax = fig.add_subplot(2,2,1)
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
ax.set_title('differenced image')
plt.imshow(differenced_image[1100:1700,:],cmap='Greys',vmax=30000,vmin=0,origin='lower')
ax = fig.add_subplot(2,2,2)
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
ax.set_title('mask')
plt.imshow(mask[1100:1700,:],cmap='Greys',vmax=1.,vmin=0,origin='lower')
ax = fig.add_subplot(2,2,3)
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
ax.set_title('dilated mask')
plt.imshow(dilated_mask[1100:1700,:],cmap='Greys',vmax=1.,vmin=0,origin='lower')
ax = fig.add_subplot(2,2,4)
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
ax.set_title('masked image')
plt.imshow(masked_image[1100:1700,:],cmap='ds9heat',vmax=20000,vmin=0,origin='lower')

inverse_dilated_mask = np.logical_not(composite_dilated_mask).astype(composite_dilated_mask.dtype)
composite = background * inverse_dilated_mask
fig = plt.figure(figsize=(14,14))
ax = fig.add_subplot(1,1,1)
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
plt.imshow(composite,cmap='ds9heat',vmax=62000,vmin=0,origin='lower')

final = composite + masked_image
fig = plt.figure(figsize=(14,14))
ax = fig.add_subplot(1,1,1)
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
plt.imshow(final,cmap='ds9heat',vmax=np.nanmax(final),vmin=0,origin='lower')

get_ipython().run_cell_magic('capture', '', "fsize = 4096.0/float(plt.rcParams['savefig.dpi'])\nfig = plt.figure(figsize=(fsize,fsize))\nax = fig.add_subplot(1,1,1)\nfig.subplots_adjust(left=0,right=1,top=1,bottom=0)\nax.get_xaxis().set_ticks([])\nax.get_yaxis().set_ticks([])\nplt.imshow(final,cmap='ds9heat',vmax=np.nanmax(final),vmin=0,origin='lower',interpolation='nearest')\nfig.savefig('mercury_composite_HMI.jpg')")

url = 'http://jsoc.stanford.edu/data/events/Mercury_MDI/WholeSun/fits/'
response = urllib.urlopen(url)
times = response.read()
files = pd.read_html(times,skiprows=2)
list_of_files = np.array(files[0][1])
print "There are",len(list_of_files),"files."

background = fits.open("http://jsoc.stanford.edu/data/events/Mercury_MDI/WholeSun/fits/20160509_170724_UTC.0148.fits")
background.verify("fix")
background = background[1].data
mask_inverse = np.ones([1024,1024])
masked_image = np.zeros([1024,1024])
composite_dilated_mask = np.zeros([1024,1024])
rangeofis = [0, 4, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120, 126, 132, 137, 142]
for i in rangeofis:
    mask = np.zeros([1024,1024])
    continuum = fits.open("http://jsoc.stanford.edu/data/events/Mercury_MDI/WholeSun/fits/"+list_of_files[i])
    continuum.verify("fix")
    continuum = continuum[1].data
    find_skewed_pixels = np.where(continuum < 170)
    for i in range(len(find_skewed_pixels[0])):
        continuum[find_skewed_pixels[0][i],find_skewed_pixels[1][i]] = 0.
    differenced_image = background - continuum
    differenced_image[0:400,:] = 0.0                # mask out the area outside the transit path
    differenced_image[500:1024,:] = 0.0
    find_mercury = np.where(differenced_image > 175)
    for i in range(len(find_mercury[0])):
        mask[find_mercury[0][i],find_mercury[1][i]] = 1.
    dilated_mask = ndimage.binary_dilation(mask, iterations=2).astype(mask.dtype)
    composite_dilated_mask = composite_dilated_mask + dilated_mask
    masked_image = continuum*dilated_mask + masked_image

inverse_dilated_mask = np.logical_not(composite_dilated_mask).astype(composite_dilated_mask.dtype)
composite = background * inverse_dilated_mask
final = composite + masked_image
fig = plt.figure(figsize=(14,14))
ax = fig.add_subplot(1,1,1)
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
plt.imshow(final,cmap='ds9heat',vmax=700,vmin=200,origin='lower')

get_ipython().run_cell_magic('capture', '', "fsize = 1024.0/float(plt.rcParams['savefig.dpi'])\nfig = plt.figure(figsize=(fsize,fsize))\nax = fig.add_subplot(1,1,1)\nfig.subplots_adjust(left=0,right=1,top=1,bottom=0)\nax.get_xaxis().set_ticks([])\nax.get_yaxis().set_ticks([])\nplt.imshow(final,cmap='ds9heat',vmax=700,vmin=200,origin='lower',interpolation='nearest')\nfig.savefig('mercury_composite_MDI.jpg')")

