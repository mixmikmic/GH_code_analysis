# Import the Google Earth Engine Python Package
import ee

# Initialize the Earth Engine object, using the authentication credentials.
ee.Initialize()

# Get the landsat 8 imagery collection
l8 = ee.ImageCollection('LANDSAT/LC8_L1T_TOA')

# Get imagery that intersects with point of interest -- San Francisco
point_of_interest = ee.Geometry.Point([-122.4371337890625, 37.724225332072436]);
spatialFiltered = l8.filterBounds(point_of_interest);

# Filter colletion by date: 2010 to present
import datetime
now = datetime.datetime.now()
d = now.day
y = now.year
m = now.month

# l8_temporalFiltered = spatialFiltered.filterDate('{}-0{}-{}'.format(y,m-1,d-2), '{}-0{}-{}'.format(y,m-1,d))

l8_temporalFiltered = spatialFiltered.filterDate('2017-01-01', '2017-02-18')

# This will sort from least to most cloudy.
sorted_collection_clouds = l8_temporalFiltered.sort('CLOUD_COVER')

# Get the first (least cloudy) image.
scene = ee.Image(l8_temporalFiltered.first())

# Band Names
scene.bandNames().getInfo()

# Parameters to visualize vegetation
vegetationParams = {'bands': 'B5,B4,B3', 'min':0, 'max': 0.3}
naturalColorParams = {'bands': 'B4,B3,B2', 'min':0, 'max': 0.3}
agricultureParams = {'bands': 'B6,B5,B2', 'min':0, 'max': 0.3}
landwaterParams = {'bands': 'B5,B6,B4', 'min':0, 'max': 0.3}
urbanParams = {'bands': 'B7,B6,B4', 'min':0, 'max': 0.3}
atmosphericParams = {'bands': 'B7,B5,B4', 'min':0, 'max': 0.3}

# Display image 
from IPython.display import Image

print("Natural Color")
Image(url=scene.getThumbUrl(naturalColorParams))

print("Land/Water")
Image(url=scene.getThumbUrl(landwaterParams))

print("Urban")
Image(url=scene.getThumbUrl(urbanParams))

print("Vegetation")
Image(url=scene.getThumbUrl(vegetationParams))

print("Agriculture")
Image(url=scene.getThumbUrl(agricultureParams))

# Metadata
scene.getInfo()



