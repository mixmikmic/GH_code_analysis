# Import libraries
from arcgis.gis import GIS
from arcgis import features
from getpass import getpass #to accept passwords in an interactive fashion
import pandas as pd

# Connect to the GIS
password = getpass()
gis = GIS("https://geosaurus.maps.arcgis.com",'arcgis_python', password)

# read the initial csv
csv1 = 'data/updating_gis_content/usa_capitals_1.csv'
cities_df_1 = pd.read_csv(csv1)
cities_df_1.head()

# print the number of records in this csv
cities_df_1.shape

# add the csv as an item
item_prop = {'title':'USA Capitals spreadsheet 2'}
csv_item = gis.content.add(item_properties=item_prop, data=csv1)
csv_item

# publish the csv item into a feature layer
cities_item = csv_item.publish()
cities_item

# update the item metadata
item_prop = {'title':'USA Capitals 2'}
cities_item.update(item_properties = item_prop, 
                   thumbnail='data/updating_gis_content/capital_cities.png')
cities_item

map1 = gis.map('USA')
map1

map1.add_layer(cities_item)

cities_item.url

# read the second csv set
csv2 = 'data/updating_gis_content/usa_capitals_2.csv'
cities_df_2 = pd.read_csv(csv2)
cities_df_2.head(5)

# get the dimensions of this csv
cities_df_2.shape

updated_df = cities_df_1.append(cities_df_2)
updated_df.shape

updated_df.drop_duplicates(subset='city_id', keep='last', inplace=True)
# we specify argument keep = 'last' to retain edits from second spreadsheet
updated_df.shape

updated_df.head(5)

import os
if not os.path.exists('data/updating_gis_content/updated_capitals_csv'):
    os.mkdir('data/updating_gis_content/updated_capitals_csv')

updated_df.to_csv('data/updating_gis_content/updated_capitals_csv/usa_capitals_1.csv')

from arcgis.features import FeatureLayerCollection
cities_flayer_collection = FeatureLayerCollection.fromitem(cities_item)

#call the overwrite() method which can be accessed using the manager property
cities_flayer_collection.manager.overwrite('data/updating_gis_content/updated_capitals_csv/usa_capitals_1.csv')

cities_flayer = cities_item.layers[0] #there is only 1 layer
cities_flayer.query(return_count_only=True) #get the total number of features

map2 = gis.map("USA")
map2

map2.add_layer(cities_item)

