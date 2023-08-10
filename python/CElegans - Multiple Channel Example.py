get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
from flowdec.nb import utils as nbutils 
from skimage import io
from scipy.stats import describe
from skimage.exposure import rescale_intensity
from flowdec import data as fd_data

acqs = fd_data.load_celegans()

acqs['CY3'].shape()

from scipy.stats import describe
for ch in acqs:
    print(ch, ': ', describe(acqs[ch].data.ravel()))

def torgb(img, dtype):
    """ Convert list of images to RGB stack """
    
    # Convert image list to array; returns (c, z, x, y)
    img = io.concatenate_images(img)
    
    # Ensure this analysis specific method is operating on the expected data type
    assert img.dtype == dtype, 'Expecting {} images, found {} instead'.format(dtype, img.dtype)
    
    # Concatenate and swap axes to (z, x, y, c)
    img = np.swapaxes(np.swapaxes(np.swapaxes(img, 0, 1), 1, 2), 2, 3)
    
    return img

img_rgb = rescale_intensity(torgb([acqs[ch].data for ch in acqs], np.uint16), out_range=(0, 255))
describe(img_rgb.ravel())

nbutils.plot_img_preview(img_rgb, zstart=50, zstop=60)

get_ipython().run_cell_magic('time', '', 'import tensorflow as tf\nfrom tfdecon import restoration as tfd_restoration\n\nniter = 200\nalgo = tfd_restoration.RichardsonLucyDeconvolver(n_dims=3).initialize()\n\nres = {ch: algo.run(acqs[ch], niter=niter) for ch in acqs}')

res_rgb = rescale_intensity(torgb([res[ch].data for ch in acqs], np.float32), out_range=(0., 1.))
describe(res_rgb.ravel())

from skimage.exposure import adjust_gamma
gamma = 1.5
def prep(img):
    img = rescale_intensity(img.max(axis=0).astype(np.float32), out_range=(0., 1.))
    return adjust_gamma(img, gamma=gamma)

fig, axs = plt.subplots(1, 2)
fig.set_size_inches((16, 16))
axs[0].imshow(prep(img_rgb))
axs[0].set_title('Original (gamma={})'.format(gamma))
axs[1].imshow(prep(res_rgb))
axs[1].set_title('Result (niter={}, gamma={})'.format(niter, gamma))
None

chs = acqs.keys()

fig, axs = plt.subplots(len(chs), 2)
fig.set_size_inches((16, 24))

for i, ch in enumerate(chs):
    axs[i][0].imshow(prep(img_rgb[:,:,:,i]))
    axs[i][0].set_title('Original ({})'.format(ch))
    axs[i][1].imshow(prep(res_rgb[:,:,:,i]))
    axs[i][1].set_title('Deconvolved ({})'.format(ch))

