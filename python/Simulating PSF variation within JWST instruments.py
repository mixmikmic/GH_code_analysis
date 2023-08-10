get_ipython().magic('matplotlib inline')
from __future__ import print_function, division
import matplotlib
matplotlib.rc('image', interpolation='nearest')
import matplotlib.pyplot as plt

import webbpsf
print("Tested with WebbPSF 0.6.0, currently running on WebbPSF", webbpsf.version.version)

miri = webbpsf.MIRI()
plt.figure(figsize=(7, 4))
miri.display()

miri.include_si_wfe

miri.pupilopd = None

plt.figure(figsize=(7, 4))
miri.display()

miri.detector_position

plt.figure(figsize=(7, 6.5))
miri_psf_center = miri.calc_psf(monochromatic=10e-6, display=True)

miri.detector_position = (10, 10)

plt.figure(figsize=(7, 4))
miri.display()

plt.figure(figsize=(7, 6.5))
miri_psf_corner = miri.calc_psf(monochromatic=10e-6, display=True)

fig, (ax_ideal, ax_realistic, ax_diff) = plt.subplots(1, 3,
                                                      figsize=(15, 3.5))
webbpsf.display_psf(miri_psf_center, ext='DET_SAMP',
                    title='MIRI Center @ 10 $\mu$m',
                    ax=ax_ideal, imagecrop=5)
webbpsf.display_psf(miri_psf_corner, ext='DET_SAMP',
                    title='MIRI Corner @ 10 $\mu$m',
                    ax=ax_realistic, imagecrop=5)
webbpsf.display_psf_difference(miri_psf_center, miri_psf_corner,
                               ext1='DET_SAMP', ext2='DET_SAMP',
                               title='Center minus Corner',
                               ax=ax_diff, cmap='RdBu_r',
                               imagecrop=5)



