get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from scipy import ndimage, signal
from flowdec import data as fd_data
from flowdec import psf as fd_psf
from flowdec import restoration as fd_restoration

actual = fd_data.neuron_25pct().data
actual.shape

kernel = np.zeros_like(actual)
for offset in [0, 1]:
    kernel[tuple((np.array(kernel.shape) - offset) // 2)] = 1
kernel = ndimage.gaussian_filter(kernel, sigma=1.)
kernel.shape

data = signal.fftconvolve(actual, kernel, mode='same')
data.shape

algo = fd_restoration.RichardsonLucyDeconvolver(data.ndim, pad_min=np.ones(data.ndim)).initialize()
res = algo.run(fd_data.Acquisition(data=data, kernel=kernel), niter=2)

res.info

get_ipython().run_cell_magic('time', '', '# Note that deconvolution initialization is best kept separate from execution since the "initialize" \n# operation corresponds to creating a TensorFlow graph, which is a relatively expensive operation and\n# should not be repeated across iterations if deconvolving more than one image\nalgo = fd_restoration.RichardsonLucyDeconvolver(data.ndim).initialize()\nres = algo.run(fd_data.Acquisition(data=data, kernel=kernel), niter=30).data')

fig, axs = plt.subplots(1, 3)
axs = axs.ravel()
fig.set_size_inches(18, 12)
center = tuple([slice(None), slice(10, -10), slice(10, -10)])
titles = ['Original Image', 'Blurred Image', 'Reconstructed Image']
for i, d in enumerate([actual, data, res]):
    img = exposure.adjust_gamma(d[center].max(axis=0), gamma=.2)
    axs[i].imshow(img, cmap='Spectral_r')
    axs[i].set_title(titles[i])
    axs[i].axis('off')

