import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import folium
import geopandas
# import json

dframe = pd.read_csv('data/innermelbourne.csv')
dframe.columns = dframe.columns.str.strip() #Taking care of formatting issues in column names.

#Dropping columns deemed irrelevant
aframe = dframe.drop(['gcc_name11','gcc_code11','sa2_5dig11','sa1_7dig11','sa3_code11','sa4_code11','ste_code11','ste_name11'],axis=1)

#Group by suburb
avg_sa2 = aframe[['sa2_name11','SumZScore']].groupby('sa2_name11').mean()

#Group by SA3
avg_sa3 = aframe[['sa3_name11','SumZScore']].groupby('sa3_name11').mean()

#To get names of all sa2 suburbs, regardless of missing values
area_suburbs = Series.sort_values(geopandas.GeoDataFrame.from_file('data/inner_melb_sa2.json')['area_name'])

#Doing this for SA1s right now.
sa1_codes = Series.sort_values(geopandas.GeoDataFrame.from_file('data/inner_melb_sa1.json')['sa1_code'])

sa3_walk_avg = avg_sa3[['SumZScore']].reset_index()
sa3_walk_avg.columns = ['SA3 Suburb','Walkability Score']
sa3_walk_avg

suburb_walk_avg = avg_sa2[['SumZScore']]
suburb_walk_avg = suburb_walk_avg.reindex(area_suburbs,fill_value=None).reset_index()
suburb_walk_avg.columns = ['SA2 Suburb','Walkability Score']
suburb_walk_avg

map_melb_sa3 = folium.Map(location=[-37.814,144.954],zoom_start = 12,max_zoom=15)

map_melb_sa3.choropleth(
    geo_path='data/inner_melb_sa3.json', #path to the geojson polygons for inner Melbourne SA3s, obtained from AURIN
    data = sa3_walk_avg, #data to bind to the choropleth
    key_on= 'properties.feature_name', #Key in the geojson file to map to the walkability scores
    columns=['SA3 Suburb', 'Walkability Score'], 
    fill_color='YlGnBu', #Colors to fill the choropleth
    line_weight=2, #Weight of the boundary line
)

map_melb_sa3.create_map('choropleth-maps/sa3melbourne.html') #Saving the maps to an html file
map_melb_sa3

map_melb_sa2 = folium.Map(location=[-37.814,144.954],zoom_start = 12,max_zoom=15)

map_melb_sa2.choropleth(
    geo_path='data/inner_melb_sa2.json',
    data = suburb_walk_avg,
    key_on= 'properties.area_name',
    columns=['SA2 Suburb', 'Walkability Score'],
    fill_color='YlGnBu',
    line_weight=2,
    legend_name = 'Walkability Score'
)

map_melb_sa2.create_map('choropleth-maps/sa2melbourne.html')
map_melb_sa2

#Code below for SA1 choropleths.
avg_sa1 = aframe[['sa1_main11','SumZScore']].groupby('sa1_main11').mean()
sa1_walk_avg = avg_sa1['SumZScore']

#Code below to set the data type of indexes to string. Important for future geoJson and choropleth operations.
sa1_walk_avg = sa1_walk_avg.reset_index()
sa1_walk_avg.columns = ['SA1 Code','Walkability Score']
sa1_walk_avg['SA1 Code'] = sa1_walk_avg['SA1 Code'].apply(str)
sa1_walk_avg.set_index('SA1 Code',inplace=True) 

sa1_walk_avg = sa1_walk_avg.reindex(sa1_codes,fill_value=None).reset_index()
sa1_walk_avg.columns = ['SA1 Code','Walkability Score']

#While building walkability, all SA1s are not considered, due to no available walkability score (explained in the report)
map_melb_sa1 = folium.Map(location=[-37.814,144.954],zoom_start = 13)

map_melb_sa1.choropleth(
    geo_path='data/inner_melb_sa1.json',
    data = sa1_walk_avg,
    key_on= 'properties.sa1_code',
    columns=['SA1 Code', 'Walkability Score'],
    fill_color='YlGnBu',
    line_weight=2,
    #threshold_scale = [-4,0,4,9,12]
)

map_melb_sa1.create_map('choropleth-maps/sa1melbourne.html')
map_melb_sa1

