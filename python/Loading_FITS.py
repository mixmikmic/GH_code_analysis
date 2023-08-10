#Define a function to handle loading of fits files given a path:
import numpy as np
import matplotlib.pyplot as plt
import pyfits as pf #(or from astropy.io import fits as pf)
get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (20,20)

def fits_load(path):
    hdu_list = pf.open(path) #loads the fits object
    image_data = hdu_list[0].data
    header = hdu_list[0].header
    return header, image_data

#Lets load a sample fits file
head, img = fits_load('2011_09_13.fits')
print 'PIXSCALE: ', head['PIXSCALE'] #example of querying the header for information
print img #show we have a 2D array of values. 

#Show the image (more on this later)
plt.imshow(img, origin='lower', cmap='gray_r',vmin=np.mean(img), vmax=np.mean(img)*2.5)
plt.show()

from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage import data
import skimage

# Compute radii in the 3rd column.

img= skimage.exposure.rescale_intensity(img, in_range=(np.min(img),np.max(img)))
#blobs_log = blob_log(img, max_sigma=30, num_sigma=10, threshold=.1)
#blobs_log[:, 2] = blobs_log[:, 2] * np.sqrt(2)
#blobs_dog = blob_dog(img, max_sigma=30, threshold=.1)
#blobs_dog[:, 2] = blobs_dog[:, 2] * np.sqrt(2)

blobs_doh = blob_doh(img,min_sigma=1,max_sigma=50)

#blobs_list = [blobs_log, blobs_dog, blobs_doh]
#colors = ['yellow', 'lime', 'red']
#titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
#          'Determinant of Hessian']
#sequence = zip(blobs_list, colors, titles)


#fig,axes = plt.subplots(1, 3, sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
#axes = axes.ravel()
#for blobs, color, title in sequence:
#    ax = axes[0]
#    axes = axes[1:]
#    ax.set_title(title)
#    ax.imshow(img, cmap='gray_r', interpolation='nearest',vmin=np.mean(img),vmax=np.mean(img)*2.0)
#    for blob in blobs:
#        y, x, r = blob
#        c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
#        ax.add_patch(c)
fig, ax = plt.subplots()
ax.imshow(img, origin='lower', cmap='gray_r',vmin=np.mean(img), vmax=np.mean(img)*2.5)
for i in range(len(blobs_doh)):
    c = plt.Circle((blobs_doh[i][1],blobs_doh[i][0]), 3*blobs_doh[i][2], linewidth=2, fill=False,color='r')
    ax.add_patch(c)
plt.show()







