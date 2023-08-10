import numpy as np
from ccdproc import CCDData, wcs_project, combine
from astropy.wcs import WCS

# To silence the VerifyWarning, do the following:
from astropy.utils.exceptions import AstropyWarning
import warnings
warnings.simplefilter('ignore', category=AstropyWarning)

#%%
# Load filename list
prefix = 'reproj_Tutorial/'
filelist = np.loadtxt(prefix+'fits.list', dtype=bytes).astype(str)

# make an empty list for original and reprojected images
before = []
reproj = []

# Specify the "target" wcs that other images to be re-projected
hdr_std = CCDData.read(prefix+filelist[0], unit='adu').header
wcs_std = WCS(hdr_std)

# Re-project all the images
for fname in filelist:
    ccd = CCDData.read(prefix+fname, unit='adu')
    before.append(ccd)
    reproj.append(wcs_project(ccd, wcs_std))

# Average combine and save
avg = combine(reproj, output_file='test_avg.fits', method='average')

from matplotlib import pyplot as plt

fig, ax = plt.subplots(1, 3, figsize=(12, 6))
im0 = ax[0].imshow(before[0].data, vmin=250, vmax=500, origin='lower')
im1 = ax[1].imshow(before[0].data/avg.data, vmin=0.9, vmax=1.2, origin='lower')
im2 = ax[2].imshow(avg.data, vmin=250, vmax=500, origin='lower')

ax[0].set_title('image 1')
ax[1].set_title('image 1 / avg_combined\n(image1 = wcs standard)', y=1.05)
ax[2].set_title('avg_combined')

plt.setp(ax[1].get_yticklabels(), visible=False)
plt.setp(ax[2].get_yticklabels(), visible=False)
plt.setp((ax[0], ax[1], ax[2]),
         xlim=[0, 1024], ylim=[0, 1024],
         xticks=np.arange(0, 1024, 250),
         yticks=np.arange(0, 1024, 250))

cbar_ax0 = fig.add_axes([0.12, 0.15, 0.23, 0.02]) # left, bottom, width, height
cbar_ax1 = fig.add_axes([0.40, 0.15, 0.23, 0.02]) # left, bottom, width, height
cbar_ax2 = fig.add_axes([0.68, 0.15, 0.23, 0.02]) # left, bottom, width, height
fig.colorbar(im0, cax=cbar_ax0, ticks=np.arange(250, 501, 50),
             orientation='horizontal', label='ADU')
fig.colorbar(im1, cax=cbar_ax1, ticks=np.arange(0.9, 1.21, 0.1),
             orientation='horizontal',
             label='ratio')
fig.colorbar(im2, cax=cbar_ax2, ticks=np.arange(250, 501, 50),
             orientation='horizontal',
             label='ADU')

plt.show()

