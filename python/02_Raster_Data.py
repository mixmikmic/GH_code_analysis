get_ipython().magic('matplotlib inline')
from matplotlib import pylab as plt

get_ipython().system('curl -o /tmp/L57.Globe.month09.2010.hh09vv04.h6v1.doy247to273.NBAR.v3.0.tiff https://data.kitware.com/api/v1/file/58dd384a8d777f0aef5d8cc2/download')

# Set the center of the map to the location the data
M.set_center(-120.32, 47.84, 7)

from geonotebook.wrappers import RasterData

rd = RasterData('file:///tmp/L57.Globe.month09.2010.hh09vv04.h6v1.doy247to273.NBAR.v3.0.tiff')
rd

M.add_layer(rd[1, 2, 3], opacity=1.0, gamma=2.5)

M.layers

print("Color   Min               Max")
print("Red:   {}, {}".format(rd[1].min, rd[1].max))
print("Green: {}, {}".format(rd[2].min, rd[2].max))
print("Blue:  {}, {}".format(rd[3].min, rd[3].max))

M.remove_layer(M.layers[0])

M.add_layer(rd[1, 2, 3], interval=(0,1))

M.remove_layer(M.layers[0])
M.add_layer(rd[1, 2, 3], interval=(0,1), gamma=0.5)

M.remove_layer(M.layers[0])
M.add_layer(rd[1, 2, 3], interval=(0,1), gamma=0.5, opacity=0.75)

# Remove the layer before moving on to the next section
M.remove_layer(M.layers[0])

M.add_layer(rd[4])

M.remove_layer(M.layers[0])

cmap = plt.get_cmap('winter', 10)
M.add_layer(rd[4], colormap=cmap, opacity=0.8)

from matplotlib.colors import LinearSegmentedColormap

M.remove_layer(M.layers[0])

# Divergent Blue to Beige to Green colormap
cmap =LinearSegmentedColormap.from_list(
  'ndvi', ['blue', 'beige', 'green'], 20)

# Add layer with custom colormap
M.add_layer(rd[4], colormap=cmap, opacity=0.8, min=-1.0, max=1.0)

M.set_center(-119.25618502500376, 47.349300631765104, 11)

layer, data = next(M.layers.annotation.rectangles[0].data)
data

import numpy as np

fig, ax = plt.subplots(figsize=(16, 16))
ax.imshow(data, interpolation='none', cmap=cmap, clim=(-1.0, 1.0))

# Adapted from the scikit-image segmentation tutorial
# See: http://scikit-image.org/docs/dev/user_guide/tutorial_segmentation.html
import numpy as np

from skimage import measure
from skimage.filters import sobel
from skimage.morphology import watershed
from scipy import ndimage as ndi


THRESHOLD = 20
WATER_MIN = 0.2
WATER_MAX = 0.6

fig, ax = plt.subplots(figsize=(16, 16))
edges = sobel(data)


markers = np.zeros_like(data)
markers[data > WATER_MIN] = 2
markers[data > WATER_MAX] = 1


mask = (watershed(edges, markers) - 1).astype(bool)
seg = np.zeros_like(mask, dtype=int)
seg[~mask] = 1

# Fill holes
seg = ndi.binary_fill_holes(seg)

# Ignore entities smaller than THRESHOLD
label_objects, _ = ndi.label(seg)
sizes = np.bincount(label_objects.ravel())
mask_sizes = sizes > THRESHOLD
mask_sizes[0] = 0

clean_segs = mask_sizes[label_objects]


# Find contours of the segmented data
contours = measure.find_contours(clean_segs, 0)
ax.imshow(data, interpolation='none', cmap=cmap, clim=(-1.0, 1.0))

ax.axis('tight')

for n, contour in enumerate(contours):
    ax.plot(contour[:, 1], contour[:, 0], linewidth=4)
  



