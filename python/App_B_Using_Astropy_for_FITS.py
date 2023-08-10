from astropy.io import fits

hdul = fits.open('HST_Tutorial/Pluto1.fits')
# hdul = HDU List = Header Data Unit List

print(type(hdul))
print()
hdul.info()
# prints out the information of the hdul.


# Let me use SCI (index 1) only
header = hdul[1].header 
data   = hdul[1].data
print('which ccd chip is this? ', header['ccdchip'])

from matplotlib import pyplot as plt
from astropy.visualization import ZScaleInterval, ImageNormalize

def znorm(image, **kwargs):
    return ImageNormalize(image, interval=ZScaleInterval(**kwargs))

def zimshow(image, **kwargs):
    plt.imshow(image, norm=znorm(image, **kwargs), origin='lower')
    plt.colorbar()
    
zimshow(data)
plt.show()

from ccdproc import CCDData, trim_image
import numpy as np

data = CCDData(data = hdul[1].data, 
               header=hdul[0].header+hdul[1].header, 
               unit='adu')

# FITS-style:
data1 = trim_image(data, fits_section='[1:200, 1:100]')

# python-style:
data2 = trim_image(data[:100, :200])
# 0:100 means 0 to 99 in Python, so that the total number is same as fits (100 pixels)

print(np.sum(data1.subtract(data2)) == 0)

# data2.write('test.fits', overwrite=True)

