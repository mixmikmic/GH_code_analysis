get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from skimage import color as sk_color
from skimage import data as sk_data
from flowdec import data as fd_data
from flowdec import restoration as fd_restoration

get_ipython().run_cell_magic('time', '', '\n# Load skimage "Astro" 2D image and generate a fake PSF\nimg = sk_color.rgb2gray(sk_data.astronaut())\npsf = np.ones((5, 5)) / 25\n\n# Add blur and noise to image\nnp.random.seed(1)\nimg_blur = fftconvolve(img, psf, \'same\') + (np.random.poisson(lam=25, size=img.shape) - 10) / 255.\n\n# Wrap image and PSF in "Acqusition" instance, which aids in doing comparisons and running\n# operations on all data associated with a data acquisition\nacquisition = fd_data.Acquisition(data=img_blur, kernel=psf)\n\n# Run deconvolution using default arguments (will default to adding no padding to image\n# as its dimensions are already powers of 2)\nimg_decon = fd_restoration.richardson_lucy(acquisition, niter=30)\n\n# Run the deconvolution again but with more care to specify how padding is done\nimg_decon_pad = fd_restoration.richardson_lucy(\n    acquisition, niter=30, \n    # Force an extra 64 pixels along each dimension\n    pad_mode=\'none\',  # Instead of default \'log2\'    \n    pad_min=[64, 64]\n)\n\n# Plot results and original\nfig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 16))\nax = ax.ravel()\nplt.gray()\n\nfor a in ax:\n       a.axis(\'off\')\n\nax[0].imshow(img)\nax[0].set_title(\'Original Image\')\n\nax[1].imshow(img_blur)\nax[1].set_title(\'Noisy Image\')\n\nax[2].imshow(img_decon)\nax[2].set_title(\'Restoration w/ No Padding\')\n\nax[3].imshow(img_decon_pad)\nax[3].set_title(\'Restoration w/ Padding\')')

plt.hist(img_decon.ravel(), log=True, bins=128, label='No Padding')
plt.hist(img_decon_pad.ravel(), log=True, bins=128, label='With Padding')
plt.gcf().set_size_inches(18, 4)
ax = plt.gca()
annotation = 'This rare but high intensity values are what make up the rings around the border'
ax.annotate(annotation, xy=(2, 10), xytext=(1.3, 1000), arrowprops=dict(facecolor='black', shrink=0.05),)
ax.set_title('Pixel Intensity Histogram')
ax.set_xlabel('Pixel Intensity')
ax.set_ylabel('Frequency')
plt.legend()
None

