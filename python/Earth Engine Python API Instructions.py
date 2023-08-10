# Import the Earth Engine Python Package into Python environment.
import ee

# Initialize the Earth Engine object, using the authentication credentials.
ee.Initialize()

# Print the information for an image asset.
image = ee.Image('srtm90_v4')
print(image.getInfo())

#celebrate!!

from IPython.display import Image
Image(url=image.getThumbUrl({'min':0, 'max': 3000}))

from geopandas import GeoDataFrame
from shapely.geometry import shape


def fc2df(fc):
    # Convert a FeatureCollection into a pandas DataFrame
   
    # Features is a list of dict with the output
    features = fc.getInfo()['features']

    dictarr = []
      
    for f in features:
        # Store all attributes in a dict
        attr = f['properties']
        # and treat geometry separately
        attr['geometry'] = f['geometry']  # GeoJSON Feature!b
        # attr['geometrytype'] = f['geometry']['type']
        dictarr.append(attr)
       
    df = GeoDataFrame(dictarr)
    # Convert GeoJSON features to shape
    df['geometry'] = map(lambda s: shape(s), df.geometry)    
    return df
# End fc2df

fc = ee.FeatureCollection("ft:1KLL3aOt7-mavHuL_uyLLPXOf7vUVk6v08XbzIepq");

#!/usr/bin/env python
#"""Select rows from a fusion table."""
import ImageTk
import Image

ee.mapclient.centerMap(-93, 40, 4)

# Select the 'Sonoran desert' feature from the TNC Ecoregions fusion table.

fc = (ee.FeatureCollection('ft:1Ec8IWsP8asxN-ywSqgXWMuBaxI6pPaeh6hC64lA')
      .filter(ee.Filter().eq('ECO_NAME', 'Sonoran desert')))

# Paint it into a blank image.
image1 = ee.Image(0).mask(0)

