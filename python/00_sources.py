import os
from urllib2 import urlopen
from zipfile import ZipFile
import geopandas as gpd

from library.get import *

# make / specify directory
dir_ = 'data/geo/countries' 
fn_zip = 'ne.zip'
dst_zip = os.path.join(dir_, fn_zip)
fn_shp = 'ne.shp'
dst_shp = os.path.join(dir_, fn_shp)
mkdir(dir_)

# make / specify directory, specify url and destination
url = 'http://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_admin_0_countries.zip'
#dst = 'data/countries/shp/ne.zip'

# download zip file
request = urlopen(url)

# save zip file
output = open(dst_zip, "w")
output.write(request.read())
output.close()

# unzip zip file
with ZipFile(dst_zip) as the_zip:
    the_zip.extractall(dir_)
    
# read in unzipped shapefile
geo = gpd.read_file(os.path.join(dir_, 'ne_10m_admin_0_countries.shp'))

# filter out countries that don't have iso a3 code
geo = geo[geo.ISO_A3 != '-99']

# reset index to iso country codes
geo.set_index('ISO_A3', inplace=True)

# drop col causing problems
geo.drop('POP_EST', inplace=True, axis=1)

# overwrite filtered list of countries
geo.to_file(dst_shp)

# pickle geodataframe
geo.to_pickle(os.path.join(dir_, 'ne.pickle'))

mkdir('data/geo/images')

# Download the data from the first table provided in the url below
# save files in data/geo/images
# unzip each tif file
# http://ngdc.noaa.gov/eog/dmsp/downloadV4composites.html

