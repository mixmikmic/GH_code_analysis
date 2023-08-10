# Set up graphing and perform imports
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

import os # for os.join to read file
import numpy as np
from scipy.ndimage import uniform_filter
import dask.array as da

# this will be our test function
def mean(img):
    "ndimage.uniform_filter with `size=51`"
    return uniform_filter(img, size=51)

get_ipython().system('ls images')

# Convert a jpeg to png with Pillow
from PIL import Image

# file names (in images folder)
imagein = os.path.join('images', 'cuba.tif')
imageout = os.path.join('images', 'cuba.png')


if not os.path.exists(imageout):
    # convert out jpeg to png
    img = Image.open(imagein)
    img.save(imageout)

# Peek at converted image
img = plt.imread(imageout)
plt.imshow(img)

# Read image data into numpy ndarray
import os

img = plt.imread(imageout)

# Take only first of 3 channels
img = (img[:,:,0] * 255).astype(np.uint8)

plt.imshow(img[::16, ::16])

# Get size and shape
mp = str(img.shape[0] * img.shape[1] * 1e-6 // 1)
'%s Mega pixels, shape %s, dtype %s' % (mp, img.shape, img.dtype)

# filter directly
get_ipython().magic('time mean_nd = mean(img)')

# mean_nd = mean(img)
plt.imshow(mean_nd[::16, ::16])

# Dask array creation - one chunk ONLY
img_da = da.from_array(img, chunks=img.shape)

# This should be about same as direct filter above - compute called to start computation
get_ipython().magic('time mean_da = img_da.map_overlap(mean, depth=0).compute()')

mean_da = img_da.map_overlap(mean, depth=0).compute()
plt.imshow(mean_da[::16, ::16])

# How many cores to we have on this computer
from multiprocessing import cpu_count
cpu_count()

img.shape, mean_da.shape, mean_nd.shape

# original pixel counts in chunk
# img_da.chunks

# Function to plot edges in original, dask chunks and difference 
#  - shows smoothness in image information boundaries from analysis
def show_edges():
    chunk_size = chunk_sizes
    size = 50
    mask = np.index_exp[chunk_size[0]-size:chunk_size[0]+size, chunk_size[1]-size:chunk_size[1]+size]
    
    # Plots
    plt.figure(figsize=(12,4))
    plt.subplot(131)
    plt.imshow(mean_nd[mask]) # filtered directly
    plt.subplot(132)
    plt.imshow(mean_da[mask]) # filtered in chunks with dask
    plt.subplot(133)
    plt.imshow(mean_nd[mask] - mean_da[mask]); # difference

# Split into 4 - so, 2x2

import math
x, y = img.shape

r = math.ceil(x / 2)
c1 = math.ceil(y / 2)

chunk_sizes = (r, c1)
print(chunk_sizes)

get_ipython().magic('time img_da = da.rechunk(img_da, chunks = chunk_sizes)')
print(img_da.chunks)

# Check to see if it's faster with multiple chunks
get_ipython().magic('time mean_da = img_da.map_overlap(mean, depth=0).compute()')

mean_da = img_da.map_overlap(mean, depth=0).compute()
plt.imshow(mean_da[::16, ::16])

show_edges()

# allow for overlapping pixels (depth > 0)
print(img_da.chunks)
get_ipython().magic('time mean_da = img_da.map_overlap(mean, depth=25).compute()')

mean_da = img_da.map_overlap(mean, depth=25).compute()
show_edges()

# Split into 16 - so, 4x4

import math
r = math.ceil(x / 4)
c1 = math.ceil(y / 4)

chunk_sizes = (r, c1)
print(chunk_sizes)

img_da = da.rechunk(img_da, chunks = chunk_sizes)
print(img_da.chunks)

get_ipython().magic('time mean_da = img_da.map_overlap(mean, depth=0).compute()')

mean_da = img_da.map_overlap(mean, depth=0).compute()
show_edges()

get_ipython().magic('time mean_da = img_da.map_overlap(mean, depth=25).compute()')

mean_da = img_da.map_overlap(mean, depth=25).compute()
show_edges()



