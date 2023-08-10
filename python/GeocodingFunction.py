import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import urllib
get_ipython().magic('matplotlib inline')

url = "https://raw.githubusercontent.com/Elixeus/PUI2015_xwang/master/plotstyle.json"
response = urllib.urlopen(url)
s = json.loads(response.read())
plt.rcParams.update(s)

import geopy
'''
Here you can import all the geocoders you like.
Use dir(geopy.geocoders) to see all the options.
'''
from geopy.geocoders import Nominatim, ArcGIS, GoogleV3, GeoNames

# read the data you want to geocode. I used my environment variable here. You can use your own.
path = os.getenv('SPATIAL')+'dataforgeocoding.csv'
data = pd.read_csv(path)

data.head()

def geocoding(fname, coder, col, n_init, n_end):
    '''
    This function takes 5 arguments: fname, coder, col, n_init, n_end
    fname: the pandas dataframe you want to geocode
    coder: geocoder of your choice
    col: the column containing the address you want to geocode
    n_init: the starting index
    n_end: the finishing index
    '''
    # create two rows: 'lat' and 'lon' for the data
    fname['lat'] = 0
    fname['lon'] = 0
    geolocator = coder
    for i in fname[col].index[n_init:n_end+1]:
        try:
            loc = geolocator.geocode(fname[col][i])
            fname.ix[i, 'lat'] = loc.latitude
            fname.ix[i, 'lon'] = loc.longitude
        except:
            print 'number %i Error' %i 
            fname.ix[i, 'lat'] = 'Error'
            fname.ix[i, 'lon'] = 'Error'
    return fname

# geocode
new = geocoding(fname = data, coder = ArcGIS(), col = 'ADDRESS',
                n_init = 0, n_end = 5)

# see the result
new.head(10)



