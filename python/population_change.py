import branca.colormap as cm
import folium
import geopandas as gpd
import json
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
# Draw matplotlib visualizations inside the Jupyter notebook
get_ipython().magic('matplotlib inline')
# Set 20 x 10 as the default size for all matplotlib visualizations
plt.rcParams['figure.figsize'] = (20, 10)
import numpy as np
import os
import pandas as pd
# Suppress scientific notation and set the precision of float values to two
pd.set_option('display.float_format', lambda x: '%.2f' % x)
# Display all columns in dataframes
pd.set_option('display.max_columns', None)
import sys

# Read the second column as a string (object) and skip the first row of column names
change = pd.read_csv('data/population_change.csv', dtype={1: 'object'}, skiprows=1)
change.head(1)

# Create a list containing the new column names
change_cols = ['id', 'id2', 'geography', 'change_1016_total', 'change_1016_natural', 'change_1016_births', 'change_1016_deaths', 'change_1016_migration', 'change_1016_international', 'change_1016_domestic', 'change_1516_total', 'change_1516_natural', 'change_1516_births', 'change_1516_deaths', 'change_1516_migration', 'change_1516_international', 'change_1516_domestic' ]
# Set the dataframe's columns equal to the list of new column names
change.columns = change_cols
change.head(1)

change_1516 = change[change.columns[[0, 1, 2, 10, 11, 12, 13, 14, 15, 16]]] # Create a new dataframe containing only the 15-16 data
change_1516.head(1)

change_1516.sort_values('change_1516_total', ascending=False).head()

change_1516.sort_values('change_1516_total', ascending=True).head()

# Read the second column as a string (object) and skip the first row of column names
pops = pd.read_csv('data/population.csv', dtype={1: 'object'}, skiprows=1)
pops.head(1)

# Create a list containing the new column names
pops_cols = ['id', 'id2', 'geography', 'pop_census', 'pop_census_base', 'pop_10', 'pop_11', 'pop_12', 'pop_13', 'pop_14', 'pop_15', 'pop_16']
# Set the dataframe's columns equal to the list of new column names
pops.columns = pops_cols
pops.head(1)

pops_1516 = pops[pops.columns[[0, 1, 2, 10, 11]]] # Create a new dataframe containing only the 15-16 data
pops_1516.head(1)

pops_change_1516 = pops_1516.merge(change_1516, on='id2')
pops_change_1516.shape # All 3,142 records matched

pops_change_1516.head(1)

pops_change_1516.drop(['id_y', 'geography_y'], axis=1, inplace=True)
pops_change_1516.rename(columns = {'id_x': 'id', 'geography_x': 'geography'}, inplace=True)
pops_change_1516.head(1)

pops_change_1516['change_1516_total_pct'] = pops_change_1516.change_1516_total / pops_change_1516.pop_15
pops_change_1516['change_1516_natural_pct'] = pops_change_1516.change_1516_natural / pops_change_1516.pop_15
pops_change_1516['change_1516_migration_pct'] = pops_change_1516.change_1516_migration / pops_change_1516.pop_15
pops_change_1516.head(1)

# Create a list containing the column names
pops_change_1516_cols = pops_change_1516.columns.tolist()
# Reorder the columns
pops_change_1516_cols = pops_change_1516_cols[:3] + [pops_change_1516_cols[4]] + [pops_change_1516_cols[3]] + [pops_change_1516_cols[5]] + [pops_change_1516_cols[-3]] + [pops_change_1516_cols[6]] + [pops_change_1516_cols[-2]] + pops_change_1516_cols[7:9] + [pops_change_1516_cols[9]] + [pops_change_1516_cols[-1]] + pops_change_1516_cols[10:12]
# Set the dataframe's columns equal to the list of reordered column names
pops_change_1516 = pops_change_1516[pops_change_1516_cols]
pops_change_1516.info()

pops_change_1516.sort_values('change_1516_total_pct', ascending=False).head()

pops_change_1516.sort_values('change_1516_total_pct', ascending=True).head()

# Define variables for each of the three change components columns
total = pops_change_1516['change_1516_total_pct']
natural = pops_change_1516['change_1516_natural_pct']
migration = pops_change_1516['change_1516_migration_pct']
plt.title('County Population Change, 2015-2016', size=36)
plt.xlabel('Percent Change', size=24)
plt.ylabel('Number of Counties', size=24)
# Plot the columns, assigning a different color to each and specifying the number of bins into which the data will be split
plt.hist([total, natural, migration], color=['green','red','blue'], alpha=0.8, bins=25)

# Define binning variables for each of the three change components columns
change_1516_total_bins = [min(pops_change_1516.change_1516_total) - 1, 0, max(pops_change_1516.change_1516_total)]
change_1516_natural_bins = [min(pops_change_1516.change_1516_natural) -1 , 0, max(pops_change_1516.change_1516_natural)]
change_1516_migration_bins = [min(pops_change_1516.change_1516_migration) - 1, 0, max(pops_change_1516.change_1516_migration)]

# Define a names variable for the groups
group_names = ['Decrease', 'Increase']

# Create the new categorical columns and add them to the dataframe
pops_change_1516['change_1516_total_category'] = pd.cut(pops_change_1516['change_1516_total'], change_1516_total_bins, labels=group_names)
pops_change_1516['change_1516_natural_category'] = pd.cut(pops_change_1516['change_1516_natural'], change_1516_natural_bins, labels=group_names)
pops_change_1516['change_1516_migration_category'] = pd.cut(pops_change_1516['change_1516_migration'], change_1516_migration_bins, labels=group_names)
pops_change_1516.head(1)

# Create a list containing the column names
pops_change_1516_cols = pops_change_1516.columns.tolist()
# Reorder the columns
pops_change_1516_cols = pops_change_1516_cols[:5] + [pops_change_1516_cols[-3]] + pops_change_1516_cols[5:7] + [pops_change_1516_cols[-2]] + pops_change_1516_cols[7:11] + [pops_change_1516_cols[-1]] + pops_change_1516_cols[11:15]
# Set the dataframe's columns equal to the list of reordered column names
pops_change_1516 = pops_change_1516[pops_change_1516_cols]
pops_change_1516.info()

pops_change_1516['change_1516_total_category'] = pops_change_1516.change_1516_total_category.astype(object)
pops_change_1516['change_1516_natural_category'] = pops_change_1516.change_1516_natural_category.astype(object)
pops_change_1516['change_1516_migration_category'] = pops_change_1516.change_1516_migration_category.astype(object)
pops_change_1516.info()

shapes = gpd.read_file('data/us_counties/us_counties.shp')
shapes.head(1)

shapes.columns = shapes.columns.str.lower()
shapes.head(1)

counties = shapes.merge(pops_change_1516, left_on='geoid', right_on='id2')
counties.shape # All 3,142 records matched

counties.head(1)

counties.plot(column='change_1516_total_category', categorical=True, cmap='RdBu', legend=True);

continental = counties[(~counties.geography.str.contains(', Alaska')) & (~counties.geography.str.contains(', Hawaii'))]
continental.plot(column='change_1516_total_category', categorical=True, cmap='RdBu', legend=True);

continental.plot(column='change_1516_natural_category', categorical=True, cmap='RdBu', legend=True);

continental.plot(column='change_1516_migration_category', categorical=True, cmap='RdBu', legend=True);

new_england = counties[(counties.geography.str.contains(', Connecticut')) | (counties.geography.str.contains(', Maine')) | (counties.geography.str.contains(', Massachusetts')) | (counties.geography.str.contains(', New Hampshire')) | (counties.geography.str.contains(', Rhode Island')) | (counties.geography.str.contains(', Vermont'))] 
new_england.shape # 67 counties returned, as expected

new_england.plot(column='change_1516_total_category', cmap='RdBu', legend=True);

# Export the new_england dataframe as a geojson file (the file type required for Folium maps)
new_england.to_file('data/new_england.geojson', driver='GeoJSON')

# Set a variable equal to the newly created geojson file
new_england_geo = r'data/new_england.geojson'
# Create a new dataframe to hold the geojson file's data
new_england_data = gpd.read_file('data/new_england.geojson')

total_map = folium.Map(location=[44.40, -70.99], zoom_start=6)
total_map.choropleth(geo_path=new_england_geo,
                          data_out='data.json',
                          data=new_england_data,
                          columns=['geoid', 'change_1516_total_pct'],
                          key_on='feature.properties.geoid',
                          fill_color='RdBu',
                          fill_opacity=0.7,
                          line_opacity=0.3,
                          legend_name='Total 2015-2016 Population Change',
                          highlight=True)
# Call the map
total_map

natural_map = folium.Map(location=[44.40, -70.99], zoom_start=6)
natural_map.choropleth(geo_path=new_england_geo,
                          data_out='data.json',
                          data=new_england_data,
                          columns=['geoid', 'change_1516_natural_pct'],
                          key_on='feature.properties.geoid',
                          fill_color='RdBu',
                          fill_opacity=0.7,
                          line_opacity=0.3,
                          legend_name='2015-2016 Population Change from Births and Deaths',
                          highlight=True)
# Call the map
natural_map

migration_map = folium.Map(location=[44.40, -70.99], zoom_start=6)
migration_map.choropleth(geo_path=new_england_geo,
                          data_out='data.json',
                          data=new_england_data,
                          columns=['geoid', 'change_1516_migration_pct'],
                          key_on='feature.properties.geoid',
                          fill_color='RdBu',
                          fill_opacity=0.7,
                          line_opacity=0.3,
                          legend_name='2015-2016 Population Change due to Migration',
                          highlight=True)
# Call the map
migration_map

total_map.save('total.html')
natural_map.save('natural.html')
migration_map.save('migration.html')



