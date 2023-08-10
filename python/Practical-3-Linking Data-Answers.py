import os
import shapely
import zipfile
import numpy as np
import pysal as ps
import pandas as pd
import geopandas as gpd
import seaborn as sns
from shapely.geometry import Point
from pysal.contrib.viz import mapping as maps # For maps.plot_choropleth

import random 
random.seed(123456789) # For reproducibility

# Make numeric display a bit easier
pd.set_option('display.float_format', lambda x: '{:,.0f}'.format(x))

# Make sure output is into notebook
get_ipython().magic('matplotlib inline')

from zipfile import ZipFile
import urllib
import os

# Create the data dir if it doesn't exist
if os.path.isdir('data') is not True:
    print("Creating 'data' directory...")
    os.mkdir('data')

# Configure the download
url  = 'http://www.reades.com/LDN-LSOAs.zip'
path = os.path.join("data","LDN-LSOAs.zip")

# Download
r    = urllib.urlretrieve(url, path)

# Unzip it into the data folder
z    = ZipFile(path)
m    = z.extractall("data")

# Configure the next download
url  = 'http://www.reades.com/NSSHRP_UNIT_URESPOP.zip'
path = os.path.join("data","NSSHRP_UNIT_URESPOP.zip")

# Download but don't unzip it
r    = urllib.urlretrieve(url, path)

# Configure the next download -- notice 
# that you will need to visit InsideAirBnB
# in order to check that the data hasn't been
# updated from 2016/10/13.
url  = 'http://data.insideairbnb.com/united-kingdom/england/london/2016-10-03/data/listings.csv.gz'
path = os.path.join("data","listings.csv.gz")

# Download but don't unzip it
r    = urllib.urlretrieve(url, path)

df       = pd.read_csv(os.path.join('data','listings.csv.gz'))
sample   = df.sample(frac=0.1)
geometry = [Point(xy) for xy in zip(sample.longitude, sample.latitude)]
crs      = {'init': 'epsg:4326'} # What projection is lat/long?
sdf      = gpd.GeoDataFrame(sample, crs=crs, geometry=geometry)
sdf      = sdf.to_crs({'init': 'epsg:27700'}) # Reproject into OSGB

# Check the output
sdf.head(3)[['id','host_id','neighbourhood','price','geometry']]

lsoas = gpd.read_file(os.path.join('data','LDN-LSOAs.shp'))
lsoas.sample(3)

z = zipfile.ZipFile(os.path.join('data','NSSHRP_UNIT_URESPOP.zip'))

nssec = pd.read_csv(z.open('Data_NSSHRP_UNIT_URESPOP.csv'), skiprows=[1])

# If we find this column, this deletes it
if np.where(nssec.columns.values=='Unnamed: 15')[0]:
    del nssec['Unnamed: 15']
if np.where(nssec.columns.values=='GEO_TYPE')[0]:
    del nssec['GEO_TYPE']
if np.where(nssec.columns.values=='GEO_TYP2')[0]:
    del nssec['GEO_TYP2']

# Initialise column names
colnames = ['CDU','GeoCode','GeoLabel','Total']
# And deal with remainder of groups
for i in range(1,9):
    colnames.append('Group' + str(i))
colnames.append('NC')
nssec.columns = colnames

# Check the output
nssec.head(3)

shpdf = pd.merge(lsoas, nssec, left_on='lsoa11cd', right_on='GeoCode', how='left')
print("Merged df is of type: " + str(type(shpdf)))

shpdf = lsoas.merge(nssec, left_on='lsoa11cd', right_on='GeoCode')
print("Shape df is of type: " + str(type(shpdf)))

print("lsoas has {0} rows, {1} columns".format(lsoas.shape[0], lsoas.shape[1]))
print("nssec has {0} rows, {1} columns".format(nssec.shape[0], nssec.shape[1]))
print("shpdf has {0} rows, {1} columns".format(shpdf.shape[0], shpdf.shape[1]))
shpdf.sample(3)

shpdf['Group1Lq'] = (
    shpdf.Group1.values / shpdf.Total.astype(float).values) / (float(shpdf.Group1.sum()) / shpdf.Total.sum()
)

shp_link = os.path.join('shapes','LDN-LSOAs.shp')
shpdf.to_file(shp_link)
values = np.array(ps.open(shp_link.replace('.shp','.dbf')).by_col('Group1Lq'))

types = ['classless', 'unique_values', 'quantiles', 'equal_interval', 'fisher_jenks']
for typ in types:
    maps.plot_choropleth(shp_link, values, typ, title=typ.capitalize(), figsize=(8,7))

# You might want to investigate what I'm up to with str.replace(...)
lsoas['Borough'] = lsoas.lsoa11nm.str.replace('\d\d\d\w$', '', case=False)
lsoas.groupby('Borough').size()

print("AirBnB CRS: " + str(sdf.crs))
print("LSOAs CRS: " + str(lsoas.crs))

lsoas = lsoas.to_crs( {'init': 'epsg:27700'} )
airbnb = gpd.sjoin(sdf, lsoas, how="inner", op='within')
print("airbnb has {0} rows, {1} columns".format(airbnb.shape[0], airbnb.shape[1]))
airbnb.sample(3)[['id','name','neighbourhood','lsoa11nm','Borough']] # Remember Borough is from the LSOA Name

# Tidy up the two strings so that they're more likely to match
airbnb['neighbourhood'] = airbnb.neighbourhood.str.strip()
airbnb['Borough'] = airbnb.Borough.str.strip()
# Are there any non-matching rows?
airbnb[ (airbnb.neighbourhood.str.strip() != airbnb.Borough.str.strip()) & (airbnb.neighbourhood.notnull()) ][
    ['id','name','neighbourhood','Borough']]

# Strip out 'LB of' and 'RB of'
airbnb['neighbourhood'] = airbnb.neighbourhood.str.replace('LB of ','').str.replace('RB of','')

# And repeat
airbnb[ (airbnb.neighbourhood.str.strip() != airbnb.Borough.str.strip()) & (airbnb.neighbourhood.notnull()) ].groupby(['Borough','neighbourhood']).size()

# Note: this is the full data set
kendf = df[ df.name.str.contains('Kensington').fillna(False) ]
print("kendf has {0} rows, {1} columns".format(kendf.shape[0], kendf.shape[1]))

# Fix the price data
kendf['price'] = kendf.loc[:,('price')].str.replace("$",'').str.replace(",",'').astype(float)

# Show what we've got
kendf.groupby('neighbourhood').size()

# Create a geodata frame from the Kensington data frame
geometry = [Point(xy) for xy in zip(kendf.longitude, kendf.latitude)]
crs      = {'init': 'epsg:4326'} # What projection is lat/long?
ken_sdf  = gpd.GeoDataFrame(kendf, crs=crs, geometry=geometry)
ken_sdf  = ken_sdf.to_crs({'init': 'epsg:27700'}) # Reproject into OSGB
#ken_sdf.sample(3)
print("Kensington listings geodata frame created...")

# Extract the Kensington LSOAs
borough_sdf = lsoas[lsoas.Borough.str.contains('Kensington')]
#borough_sdf.sample(3)
print("Kensington borough geodata frame created...")

# Create the output dir if it doesn't exist
if os.path.isdir('output') is not True:
    print("Creating 'output' directory...")
    os.mkdir('output')

borough_sdf.to_file(os.path.join('output','Kensington.shp'))
ken_sdf.to_file(os.path.join('output','KensingtonListings.shp'))
print("Shapefiles written.")

import math
my_bbox = list(ken_sdf.total_bounds)
my_bbox[0] = int(math.floor(my_bbox[0] / 100.0)) * 100
my_bbox[1] = int(math.floor(my_bbox[1] / 100.0)) * 100
my_bbox[2] = int(math.ceil(my_bbox[2] / 100.0)) * 100
my_bbox[3] = int(math.ceil(my_bbox[3] / 100.0)) * 100
print(my_bbox)

from pylab import *

the_borough = ps.open(os.path.join('output','Kensington.shp'))
listings = ps.open(os.path.join('output','KensingtonListings.shp'))

listingsdata = ps.open(os.path.join('output','KensingtonListings.dbf'))
print("There are " + str(len(listingsdata)) + " rows in the listings DBF.")

prices = np.array(listingsdata.by_col['price']) # Retrieve pricing data
pricesq5 = ps.esda.mapclassify.Quantiles(prices, k=5) # Classify into 5 quantiles
print(pricesq5) # Show the classification result

fig = figure(figsize=(7,13)) # Why do you think I changed the figure size this way?

base = maps.map_poly_shp(the_borough)
base.set_facecolor('none')
base.set_linewidth(0.5)
base.set_edgecolor('0.8')

lyr = maps.map_point_shp(listings)
lyr = maps.base_choropleth_classif(lyr, pricesq5.yb, cmap='inferno')
lyr.set_alpha(0.5)
lyr.set_linewidth(0.)
lyr.set_sizes(np.repeat(10, len(listingsdata))) # Sice of the dots

plt.title("Quantile Classification of AirBnB Sample")
ax = maps.setup_ax([base, lyr], [my_bbox, my_bbox]) # You should get a 'crop' of London
fig.add_axes(ax)
show() # There is one right under the 'o' of Classification
       # You will need to scroll down to the map!
print("Done.")



