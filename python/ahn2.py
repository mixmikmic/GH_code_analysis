# we'll be using these services
import owslib.wms
import owslib.wcs
import owslib.wfs
from io import BytesIO, StringIO
import os
import json
import shapely.geometry

import matplotlib.style
matplotlib.style.use('ggplot')
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# this is the dataset of dutch altimetry
url = 'http://geodata.nationaalgeoregister.nl/ahn2/wms'

# we're using WMS for rendered maps
wms = owslib.wms.WebMapService(url)

# let's see what the server offers.
wms.contents

# let's zoom in on delft
(top,left, bottom, right) = (52.0144342,4.3053329,51.984102,4.3947685)
# define a bounding box
bbox = (left, bottom, right, top) 
# and get a map
f = wms.getmap('ahn2_05m_ruw', bbox=bbox, size=(256, 256))

# oops, we missed some inputs
import logging
logging.warn(f.read())

# let's specify some more details, 
# We want WGS84 and we want a bigger map
f = wms.getmap(['ahn2_5m'], srs='EPSG:4326', bbox=bbox, size=(1024, 1024), format='image/png', transparent=True)

# convert binary data to IO stream
f_io = BytesIO(f.read())
img = plt.imread(f_io)

plt.subplots(figsize=(13,8))
plt.imshow(img)

# let's find out some more details about the server
print(wms.getServiceXML().decode('ascii'))

# let's zoom out
(top,left, bottom, right) = (53.8,3,50.5,7.1)
bbox = (left, bottom, right, top) 
f = wms.getmap(['ahn2_5m'], srs='EPSG:4326', bbox=bbox, size=(1024, 1024), format='image/png', transparent=True)
f_io = BytesIO(f.read())
img = plt.imread(f_io)
plt.imshow(img)

(top,left, bottom, right) = (52.05,4.30,51.95,4.40)
bbox = (left, bottom, right, top) 

f = wms.getmap(['ahn2_5m'], srs='EPSG:4326', bbox=bbox, size=(1024, 1024), format='application/vnd.google-earth.kml', transparent=True)
open('file.kml', 'w').write(f.read().decode('ascii'))

os.system('file.kml')

url='http://geodata.nationaalgeoregister.nl/ahn2/wcs'
wcs = owslib.wcs.WebCoverageService(url=url, 
                                    version="1.0.0") # version is actually 1.1.1, 1.2.0 
meta = wcs.contents['ahn2:ahn2_5m']
bbox = meta.boundingBoxWGS84

# How do we get this to work?
f = wcs.getCoverage(identifier='ahn2:ahn2_5m',
                bbox=bbox, 
                format='GeoTIFF',
                crs='EPSG:28992', 
                version="1.0.0",
                resx=5, resy=5)
f.read()[1:1000]

url = 'http://geodata.nationaalgeoregister.nl/ahn2/wfs'
wfs = owslib.wfs.WebFeatureService(url, version="2.0.0")
# only one layer
wfs.contents
print(wfs)

layer = wfs.contents['ahn2:ahn2_bladindex']
layer = list(wfs.contents.values())[0]

# let's read the features
f = wfs.getfeature(typename=[layer.id], outputFormat="json")

data = json.loads(f.read().decode('ascii'))

shapes = []
for feature in data['features']:
    shapes.append(shapely.geometry.asShape(feature['geometry'])[0])

import numpy as np
for shape in shapes:
    xy = np.array(shape.exterior.coords)
    plt.plot(xy[:,0], xy[:,1], 'k-')

data['features'][0]

# Challenge
# read point cloud from http://geodata.nationaalgeoregister.nl/ahn2/atom/ahn2_uitgefilterd.xml



