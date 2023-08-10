# Tell Jupyter to show images in this browser window Matlab style instead of popping out
get_ipython().magic('matplotlib inline')

# debugging flags. Reloads include files every execution.
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

import pprint
pp = pprint.PrettyPrinter(indent=4)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Import the Google Earth Engine Library
import ee

# And all my helper functions:
from gee_library import *

# This command initializes the library and connects to Google's servers.
ee.Initialize()

# Now we'll load up some satelite imagery of Monterey, collected by the USDA Farm Service Agency.
# Don't worry too much about the details; just make sure an image appears below to confirm
# that your installation is working.
nps_bounds = bound_geometry(
                        (-121.880742, 36.600885),
                        (-121.866788, 36.594170))
tiles = img_at_region(ee.ImageCollection('USDA/NAIP/DOQQ'), 5, ['R'], nps_bounds)
plt.imshow(tiles['R'], cmap='gray')
plt.show()


# First we create load a collection from the GEE database. 
world_collection = ee.ImageCollection('USDA/NAIP/DOQQ')

# An ee.ImageCollection is a collection of images. Each image contains certain spectrum bands. The
# world_collection we just created has thousands of images included in it; operating on such a large
# collection takes a long time. But we aren't interested in the whole world. We're just interested
# in a small area. Let's pare down the collection.

# We have to create an official ee.Geomtery object to communicate to Google Earth Engine the
# spatial bounds that we are interested in. I created a helper (factory?) function that creates 
# this object. 
nps_bounds = bound_geometry(
                        (-121.880742, 36.600885),
                        (-121.866788, 36.594170))


# We can use that object to apply a filter to the world_collection to only include images
# that intersect the spatial bounds we define.   
monterey_collection = world_collection.filterBounds(nps_bounds)

# And then only select images taken in 2016. The first date given to ee.Filter.date is inclusive, the
# second date is exclusive (much like Python's range function).
monterey_collection_2016 = monterey_collection.filter(ee.Filter.date('2016-01-01', '2017-01-01'))


print collection_length(monterey_collection_2016), "images available in monterey_collection."

# Ok, so 2 images were taken in 2016. Which dates were these images taken? I've created a helper function to query
# which dates are available in a collection.
dates_of_images = dates_available(monterey_collection_2016)
print "Dates available:",dates_of_images

# I also created a function to query which bands are available in a collection.
available_bands_in_monterey = available_bands(monterey_collection_2016)
for k, v in available_bands_in_monterey.items():
    print "Band", k, "is available in", v['number_available'], "images. (", v['percent_available'], "%)"
    
# This imagery is available in 4 bands: R (red), G (green), B(blue), and N (near-IR)

# When loading imagery, we have to decide which resolution we want, measured in meters per pixel.
# I created a helper function that, given a geometry bound box and resolution, will estimate the
# image size that will result. (Pulling too large of an image will often result in error/failure.)
# Note that this is very rough estimate and has some projection issues right now.
estimated_size = estimate_image_size_at_resolution(nps_bounds, 5)
print "At 5 meters/pixel, nps_bounds measures", estimated_size

# Load the most recent imagery
print "Imagery of NPS, taken at 2016-07-12"
new_monterey_collection = monterey_collection.filter(ee.Filter.date('2016-07-12'))
tiles = img_at_region(new_monterey_collection, 5, 'R', nps_bounds)
plt.imshow(tiles['R'], cmap='gray'); plt.show()

# It looks like the imagery from that day only includes part of our FOV.
# Let's also include the second-most recent image, and let Google fuse them together. (Remember,
# the second date in the range is exclusive.)
print "Fusing images from 2016-06-19 and 2016-07-12 by specifying the range 2016-06-19 - 2016-07-13"
new_monterey_collection = monterey_collection.filter(ee.Filter.date('2016-06-19', '2016-07-13'))
tiles_combined = img_at_region(new_monterey_collection, 5, 'R', nps_bounds)
plt.imshow(tiles_combined['R'], cmap='gray'); plt.show()

# Let's compare this with the earliest snapshots.
print "Imagery from 2005"
old_monterey_collection = monterey_collection.filter(ee.Filter.date('2005-01-01', '2006-01-01'))
tiles_combined = img_at_region(old_monterey_collection, 5, 'R', nps_bounds)
plt.imshow(tiles_combined['R'], cmap='gray'); plt.show()


# We'll import numpy to help with the matrix operations.
import numpy as np

# Let's download the bands in the visible spectrum.
tiles = img_at_region(new_monterey_collection, 3, ['R','G', 'B'], nps_bounds)

# Numpy can stack 2 dimentional images in a 3rd dimention using the command dstack. Note that the output of the command
# is a numpy array, not a traditional python List. This is a detail that won't affect us here but might trip you
# up if your script calls List specific functions like len().
img = np.dstack((tiles['R'], tiles['G'], tiles['B']))

# We plot it the same way, but the cmap parameter is uneccessary since matplotlib will use true colors.
plt.imshow(img); plt.show()

print img.shape

import scipy.misc


nps_center = [ -121.873925, 36.596853]


# I've created a function that calculates a square geometry with equal height and width in meters. Since the
# function uses a radius to calculate distance, it requires a half_distance parameter, which defines half the
# length of each side of the square. Below we request a geometry that desribes a square patch of land, centered
# at NPS, measureing 900 meters on easch side.
tile_bounds = square_centered_at(
    point = nps_center,
    half_distance = 450
)

# We can use that geometry to request imagery the same way as before. Since our patch is 900 meters on every side,
# requesting a resolution of 3 meters per pixel should result in an image that is 300x300 pixels.
tiles = img_at_region(monterey_collection, 3, ['R'], tile_bounds)

# We'll convert the image into a Numpy array and look at the dimentions.
img = np.array(tiles['R'])
print "The image has dimentions", img.shape,"; we expected (300, 300)"

# Hmm, I don't know why the width always overshoots. Probably something to do with the projection.
# We'll fix that (as well as the pixel or two difference in the other axis) by resizing using bilinear interpolation.
img = scipy.misc.imresize(img, (300, 300))
print "After interpolation, the image has dimentions", img.shape

plt.imshow(img, cmap='gray'); plt.show()

