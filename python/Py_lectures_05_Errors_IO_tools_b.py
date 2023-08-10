import os
import numpy as np
import matplotlib.pylab as plt
get_ipython().magic('matplotlib inline')

# Masked arrays
import numpy.ma as ma

# The file below contains a noise signal, made of a sequence of spikes (called Barkhausen jumps)
# The first line is the total time of the signal
# Note: the signal is negative, for simplicity use -signal
filename = "/home/gf/src/Python/Python-in-the-lab/Bk/F64ac_0.03_time_200.dat"
data = -np.loadtxt(filename)
with open(filename) as f:
    header = f.readline()
max_time = float(header[1:])
time = np.linspace(0, max_time, len(data))
fig = plt.figure(figsize=(18,8))
plt.plot(time, data, 'k')
data_mean = np.mean(data)
plt.plot(time, data_mean * np.ones_like(time), '--r', lw=2)

# ahah, what if we want to consider just the values above the average, and calculate the new mean?
# mask = data < data_mean
# data_masked = ma.masked_array(data, mask)
data_masked = ma.masked_less_equal(data, data_mean)
new_mean = data_masked.mean()
plt.plot(time, new_mean * np.ones_like(time), '-r', lw=2);

data_masked[160:230]

# It would be nice to plot the data above the original mean
# i.e. cut the data below it
fig = plt.figure(figsize=(18,8))
plt.plot(time, data, 'k')
plt.plot(time, data_masked.filled(data_mean), '-r', lw=2);

import sympy as sp
sp.init_printing()
from IPython.display import display

x = sp.symbols("x")
f = sp.sin(x)
display(f, sp.diff(f, x))
g = sp.diff(sp.sin(x)*sp.exp(-x), x)
display(g)
g_int = sp.integrate(g, x)
display(g_int)

out = sp.integrate(sp.sin(x**2), (x, -sp.oo, sp.oo))
display(out)
# Not bad...

import theano
from theano import tensor

# declare two symbolic floating-point scalars
a = tensor.dscalar()
b = tensor.dscalar()

# create a simple expression
c = a + b

# convert the expression into a callable object that takes (a,b)
# values as input and computes a value for c
f = theano.function([a,b], c)

# bind 1.5 to 'a', 2.5 to 'b', and evaluate 'c'
assert 4.0 == f(1.5, 2.5)

# From http://scikit-image.org/docs/stable/auto_examples/segmentation/plot_random_walker_segmentation.html
from skimage.segmentation import random_walker
from skimage.data import binary_blobs
import skimage

# Generate noisy synthetic data
data = skimage.img_as_float(binary_blobs(length=256, blob_size_fraction=0.15, volume_fraction=0.3, seed=132))
data += 0.35 * np.random.randn(*data.shape)
markers = np.zeros(data.shape, dtype=np.uint)
markers[data < -0.3] = 1
markers[data > 1.3] = 2

# Run random walker algorithm
labels = random_walker(data, markers, beta=10, mode='bf')

# Plot results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6),
                                    sharex=True, sharey=True)
ax1.imshow(data, cmap='gray', interpolation='nearest')
ax1.axis('off')
ax1.set_adjustable('box-forced')
ax1.set_title('Noisy data')
ax2.imshow(markers, cmap='hot', interpolation='nearest')
ax2.axis('off')
ax2.set_adjustable('box-forced')
ax2.set_title('Markers')
ax3.imshow(labels, cmap='gray', interpolation='nearest')
ax3.axis('off')
ax3.set_adjustable('box-forced')
ax3.set_title('Segmentation')

fig.tight_layout()
plt.show()

import tifffile
from libtiff import TIFF # Unfortunately this works only in Python 2.X

class Images:
    def __init__(self, root_dir, pattern, resolution=np.int16):
        filename = os.path.join(root_dir, pattern)
        self.images = self._from_tif(filename, resolution)
        
    def _from_tif(self, filename, resolution):
        try:
            print("Loading %s" % filename)
            with tifffile.TiffFile(filename) as tif:
                frames = tif.micromanager_metadata['summary']['Frames']
                height = tif.micromanager_metadata['summary']['Height']
                width = tif.micromanager_metadata['summary']['Width']
                max_gray_level = tif.micromanager_metadata['display_settings'][0]['Max']
                bit_depth = tif.micromanager_metadata['summary']['BitDepth']
                images = tif.asarray()
            images = images.astype(self.resolution)
        except UnboundLocalError as e:
            print("The error is: %s" % e)
            print("Cannot load the %s file, try using libtiff (slower)" % filename)
            print("frames: %i, size: (%i,%i), bit depth: %i, max of gray level %i" % (frames, height, width, bit_depth, max_gray_level))
            tif = TIFF.open(filename, mode='r')
            images = np.empty((frames, height, width)).astype(resolution)
            for i,image in enumerate(tif.iter_images()):
                images[i] = image
            tif.close()
        return images

mainDir = "/home/gf/src/Python/Python-in-the-lab/images"
filename = "02_Irr_800uC_0.232A_MMStack_Pos0.ome.tif"

im = Images(mainDir, filename)

im0 = im.images[0]
plt.imshow(im0, 'gray')

import skimage.exposure as expo
plt.imshow(expo.equalize_hist(im0), 'gray')

im.images.shape

im100 = im.images[100]
fig,axs = plt.subplots(1, 3, figsize=(18,8))
for ax in axs: 
    ax.set_axis_off()
r0, r1, c0, c1 = 230, 840, 205, 805
axs[0].imshow(expo.equalize_hist(im0[r0:r1,c0:c1]), 'gray')
axs[1].imshow(expo.equalize_hist(im100[r0:r1,c0:c1]), 'gray')
axs[2].imshow(expo.equalize_hist(im100[r0:r1,c0:c1] - im0[r0:r1,c0:c1]), 'gray');

from IPython.display import Image
imageDir = "/home/gf/src/Python/Python-in-the-lab/images"
Image(filename=os.path.join(imageDir,"02_Irr_16e8He_0.232_700ms.jpg"))

try:
    import pycuda.autoinit
    import pycuda.driver as drv
    from pycuda.compiler import SourceModule

    
    mod = SourceModule("""
__global__ void multiply_them(float *dest, float *a, float *b)
{
  const int i = threadIdx.x;
  dest[i] = a[i] * b[i];
}
""")

    multiply_them = mod.get_function("multiply_them")

    a = numpy.random.randn(400).astype(numpy.float32)
    b = numpy.random.randn(400).astype(numpy.float32)

    dest = np.zeros_like(a)
    multiply_them(
        drv.Out(dest), drv.In(a), drv.In(b),
        block=(400,1,1), grid=(1,1))

    print dest-a*b
except:
    print("pyCuda not installed")

import pickle

# Let's save the masked array we made at the beginning
print(data_masked.shape)
with open("masked_array.pkl", 'w') as f:
    pickle.dump(data_masked, f)

# Let's check it if we really wrote the file
import glob
glob.glob1(".", "masked*")

# Try to unpickle
with open("masked_array.pkl", 'rb') as pickle_file:
    m_array = pickle.load(pickle_file)

m_array[450:480]

data_masked[450:480]

