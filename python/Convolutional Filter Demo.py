import numpy as np
import scipy
from scipy import misc
from scipy import signal
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

f = np.array([3,2,1])
g = np.array([1, 2, 3, 4, 5])

np.convolve(f, g, mode='valid')

def gkern(l=5, sig=1.):
    """
    creates gaussian kernel with side length l and a sigma of sig
    """

    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2) / (2. * sig**2))

    return kernel / np.sum(kernel)

f =  gkern(l=21, sig=10)
print(f.shape)

g = scipy.misc.imread('dirks.jpg', 'F')
print(g.shape)

plt.imshow(g)
plt.set_cmap('gray')
plt.gcf().set_size_inches((10, 10))

blurry =scipy.signal.convolve2d(g, f, mode='valid')
plt.imshow(blurry)
plt.set_cmap('gray')
plt.gcf().set_size_inches((10, 10))

def edge_detector(size, vertical=True):
    a = np.arange(-(size // 2), size // 2 + 1)
    f = np.tile(a, (len(a), 1))
    if vertical:
        return f
    return f.T

f = edge_detector(3)
print(f)

g = scipy.misc.ascent()
plt.imshow(g)
plt.set_cmap('gray')
plt.gcf().set_size_inches((10, 10))

edge = scipy.signal.convolve2d(g, f, mode='valid')
plt.imshow(edge)
plt.set_cmap('gray')
plt.gcf().set_size_inches((10, 10))

f = edge_detector(3, vertical=False)
print(f)

edge = scipy.signal.convolve2d(g, f, mode='valid')
plt.imshow(edge)
plt.set_cmap('gray')
plt.gcf().set_size_inches((10, 10))



