import os.path as op
import dipy.data as dpd
remote, local = dpd.fetch_stanford_t1()
t1_file = op.join(local, 't1.nii.gz')

import nibabel as nib

t1_img = nib.load(t1_file)
t1_data = t1_img.get_data()

from skimage import exposure
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

hist, bin_centers = exposure.histogram(t1_data.ravel())
fig, ax = plt.subplots(1)
ax.plot(bin_centers, hist)

import numpy as np
t1_norm = (t1_data - np.mean(t1_data))/np.std(t1_data)

hist, bin_centers = exposure.histogram(t1_norm.ravel())
fig, ax = plt.subplots(1)
ax.plot(bin_centers, hist)

# Normalize the histogram to sum to 1:
hist = hist.astype(float) / np.sum(hist)

# class probabilities for all possible thresholds
weight1 = np.cumsum(hist)
weight2 = np.cumsum(hist[::-1])[::-1]

#Plotting this:
fig, ax = plt.subplots(1)
ax.plot(bin_centers, weight1)
ax.plot(bin_centers, weight2)

# class means for all possible thresholds
mean1 = np.cumsum(hist * bin_centers) / weight1
mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]

#Plotting this:
fig, ax = plt.subplots(1)
ax.plot(bin_centers, mean1)
ax.plot(bin_centers, mean2)

# The last value of `weight1`/`mean1` should pair with zero values in
# `weight2`/`mean2`, which do not exist.
variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

fig, ax = plt.subplots(1)
ax.plot(bin_centers[:-1] ,variance12)

idx = np.argmax(variance12)
threshold = bin_centers[:-1][idx]

fig, ax = plt.subplots(1)
ax.plot(bin_centers, hist)
ax.plot([threshold, threshold], [0, ax.get_ylim()[1]])

binary = t1_norm >= threshold
fig, ax = plt.subplots()
ax.matshow(binary[:, :, binary.shape[-1]//2], cmap='bone')

from skimage import filters
for threshold in [filters.threshold_isodata, 
                  filters.threshold_li,
                  filters.threshold_otsu, 
                  filters.threshold_yen]:

    fig, ax = plt.subplots(1)
    th = threshold(t1_data[:, :, t1_data.shape[-1]//2])
    binary = t1_data >= th
    ax.matshow(binary[:, :, binary.shape[-1]//2], cmap='bone')

from skimage import feature

im = t1_data[:, :, t1_data.shape[-1]//2]

edges1 = feature.canny(im, sigma=1)
edges7 = feature.canny(im, sigma=7)
fig, ax = plt.subplots(1, 2)
ax[0].matshow(edges1, cmap='bone')
ax[1].matshow(edges7, cmap='bone')

from scipy import ndimage as ndi
dilated = ndi.binary_dilation(edges7, iterations=4)
fill_brain = ndi.binary_fill_holes(dilated)

fig, ax = plt.subplots(1)
ax.matshow(fill_brain, cmap='bone')

brain = np.zeros(im.shape)
brain[fill_brain] = im[fill_brain]
fig, ax = plt.subplots(1)
ax.matshow(brain, cmap='bone')

brain = np.copy(im)
brain[fill_brain] = 0
fig, ax = plt.subplots(1)
ax.matshow(brain, cmap='bone')

from scipy import ndimage as ndi
import matplotlib.pyplot as plt

from skimage.morphology import watershed, disk
from skimage import data
from skimage.filters import rank
from skimage.util import img_as_ubyte

image = np.zeros(im.shape)
th_otsu = filters.threshold_otsu(im)
image[im>th_otsu] = 1

# find continuous region (low gradient -
# where less than 10 for this image) --> markers
# disk(5) is used here to get a more smooth image
markers = rank.gradient(image, disk(5)) < 10
markers = ndi.label(markers)[0]

# local gradient (disk(2) is used to keep edges thin)
gradient = rank.gradient(image, disk(2))

# process the watershed
labels = watershed(gradient, markers)

# display results
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), sharex=True, 
                         sharey=True, subplot_kw={'adjustable':'box-forced'})
ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
ax[0].set_title("Original")

ax[1].imshow(gradient, cmap=plt.cm.hot, interpolation='nearest')
ax[1].set_title("Local Gradient")

ax[2].imshow(markers, cmap=plt.cm.spectral, interpolation='nearest')
ax[2].set_title("Markers")

ax[3].imshow(im, cmap=plt.cm.gray, interpolation='nearest')
ax[3].imshow(labels, cmap=plt.cm.spectral, interpolation='nearest', alpha=.7)
#ax[3].imshow(labels<3, cmap=plt.cm.bone, interpolation='nearest', alpha=.7)
ax[3].set_title("Segmented")

for a in ax:
    a.axis('off')

fig.tight_layout()
plt.show()



from dipy.segment.mask import median_otsu
t1_mask, mask = median_otsu(t1_data)

fig, ax = plt.subplots(1)
ax.matshow(t1_mask[:, :, t1_mask.shape[-1]//2], cmap='bone')



