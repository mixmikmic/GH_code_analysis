import astroscrappy
print(astroscrappy.detect_cosmics.__doc__)

from matplotlib import pyplot as plt
from astropy.visualization import ZScaleInterval, ImageNormalize
from astropy.io import fits

################################################################################
# Following are functions to plot "zscale" image using astropy.visualization:
def znorm(image, **kwargs):
    return ImageNormalize(image, interval=ZScaleInterval(**kwargs))

def zimshow(image, **kwargs):
    plt.imshow(image, norm=znorm(image, **kwargs), origin='lower')
    plt.colorbar()
################################################################################

hdul1 = fits.open('HST_Tutorial/Pluto1.fits')

plt.figure(figsize=(10,10))
zimshow(hdul1[1].data)
plt.show()

import astroscrappy
from astropy.nddata.utils import Cutout2D
from astropy.stats import sigma_clipped_stats
from ccdproc import CCDData, combine
from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt


hdul = fits.open('HST_Tutorial/Pluto1.fits')

obj = CCDData(data = hdul[1].data,
              header=hdul[0].header+hdul[1].header,
              unit='adu')
# Although the image is in electrons per second unit, let me just use "adu".
# The reason for "header" setting, see appendix B.

obj_LA = obj.copy()
obj_cr = obj.copy()

# Following should give identical result to IRAF L.A.Cosmic,
# "m_LA" is the mask image
m_LA, obj_LA.data = astroscrappy.detect_cosmics(obj.data,
                                                inmask=None,
                                                satlevel=np.inf,
                                                sepmed=False,
                                                cleantype='medmask',
                                                fsmode='median',
                                                readnoise=5,
                                                objlim=4)
obj_LA.write('LA_Pluto.fits', overwrite=True)


# Following is the "fastest" version. The author argues that this
# method gave comparable result as default L.A.Cosmic, but 100 times faster.
# I personally do not prefer this.
# Also the satlevel is given as np.inf, since HST has pixel values in 
m_cr, obj_cr.data = astroscrappy.detect_cosmics(obj.data,
                                                readnoise=0,
                                                satlevel=np.inf)
obj_cr.write('cr_Pluto.fits', overwrite=True)

plt.figure(figsize=(10,10))
plt.imshow(obj.divide(obj_LA), vmin=0, vmax=3)
plt.title('original / LA cosmic (== The map of cosmic rays)')
plt.colorbar()
plt.show()

plt.figure(figsize=(10,10))
zimshow(obj_LA)
plt.title('LA cosmic (== CR rejected)')
plt.show()


plt.figure(figsize=(10,10))
plt.imshow(obj.divide(obj_LA), vmin=0, vmax=3)
plt.title('original / astroscrappy default (== The map of cosmic rays)')
plt.colorbar()
plt.show()

plt.figure(figsize=(10,10))
zimshow(obj_cr)
plt.title('astroscrappy default cosmic (== CR rejected)')
plt.show()



