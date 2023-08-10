get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.misc
import numpy as np

import pprint as pp
import ee
ee.Initialize()

from gee_library import *

import pprint as pp

SAR_dataset = ee.ImageCollection('NOAA/DMSP-OLS/NIGHTTIME_LIGHTS')
SAR_dataset = ee.ImageCollection('USDA/NAIP/DOQQ')

meters = 3000
pixels = 200
resolution = meters/pixels


nps_center = [ -121.873925, 36.596853]
laplata = [-76.915988, 38.571057]
tile_bounds = square_centered_at(
    point = laplata,
    half_distance = meters/2
)


monterey_SAR_Collection = SAR_dataset.filterBounds(tile_bounds)



print "Getting date slices..."
date_slice_list = date_slices(monterey_SAR_Collection.select(['R', 'G', 'B']), tile_bounds)


print "Done.", len(date_slice_list), "slices found. Getting imagery for each slice..."
for start_date, end_date in date_slice_list:
    
    print timestamp_to_datetime(start_date.getInfo()['value']), "through", timestamp_to_datetime(end_date.getInfo()['value']),":"
    
    filtered_collection = monterey_SAR_Collection.filter(ee.Filter.date(start_date, end_date))

    # Request imagery
    tiles = img_at_region(geCollection=filtered_collection,
                          resolution=resolution,
                          bands=['R', 'G', 'B'],
                          geo_bounds=tile_bounds)
    # resize img to requested size
    np_band_array = [scipy.misc.imresize(tiles[b], (pixels, pixels)) for b in ['R', 'G', 'B']]
    
    # and stack the images in a matrix
    img = np.dstack(np_band_array)
    
    # Display the image
    plt.imshow(img); plt.show()

