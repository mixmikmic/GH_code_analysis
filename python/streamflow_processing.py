get_ipython().run_line_magic('pylab', 'inline')
import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gp
import matplotlib as mpl
import matplotlib.pyplot as plt
from jupyterthemes import jtplot
import utm
from scipy.spatial import KDTree
jtplot.style(jtplot.infer_theme(), context='paper', fscale=2)
jtplot.figsize(x=20, y=12)
from futurefish.data_processing import calcLatLon, get_model_ts, metric_min7day_streamflow, locate_nearest_neighbor_values

SHAPEFILES = glob.glob('../../data/**/**/*.shp')
STREAMFLOW_META = '../full_site_test_dataset.csv'

dataframes = [gp.GeoDataFrame.from_file(shpfile) for shpfile in SHAPEFILES]
gdf = gp.GeoDataFrame(pd.concat(dataframes, ignore_index=True))

# Extract out the variables we want to use because it's a large dataset
# and a smaller sample will be faster to work with
gdf_selected_columns = gdf[['S39_2040DM', 'S41_2080DM', 'geometry']]

translating_temperature_keys_dictionary = {'S39_2040DM': 'Stream Temperature 2040s',
                                         'S41_2080DM':  'Stream Temperature 2080s'}

# Remove the sites with NaNs
cleaned_up_gdf = gdf_selected_columns[gdf_selected_columns['S39_2040DM'].notnull()]

# Convert the coordinates from eastings/northings to degrees longitude
# and degrees latitude
lat_lons = []
for (i, point) in enumerate(cleaned_up_gdf.geometry[:]):
    # The false easting is from streamflow temperature
    # dataset documentation within the GIS shapefile
    false_easting = 1500000 
    northing = point.coords.xy[1][0]  
    easting = point.coords.xy[0][0] - false_easting
    [lat, lon] = calcLatLon(northing, easting)
    lat_lons.append([lat, lon])
temperature_sites = np.array(lat_lons)

streamflow_sites = pd.read_csv(STREAMFLOW_META)

# Select out the sites in the United States because the temperature data
# is only available in the U.S. So, south of the 49th parallel!
streamflow_sites = streamflow_sites[streamflow_sites['Latitude'] < 49 ]

collated_dataset = pd.DataFrame(index=streamflow_sites['Site ID'], 
                                columns=list(translating_temperature_keys_dictionary.values()))
for site in streamflow_sites['Site ID']:
    
# Loop through each location in the streamflow set and
# select the 10 nearest points within the stream temperature set
    point = [streamflow_sites[streamflow_sites['Site ID']==site]['Latitude'].values[0],
             streamflow_sites[streamflow_sites['Site ID']==site]['Longitude'].values[0]]
    locate_nearest_neighbor_values(point, cleaned_up_gdf, temperature_sites)
    for variable in translating_temperature_keys_dictionary.keys():
        collated_dataset.set_value(site, 
                        translating_temperature_keys_dictionary[variable], 
                        nearest_neighbors_data[variable].mean())

streamflow_timeframes = {'Streamflow 2040s': slice('2029-10-01', '2059-09-30'),
                        'Streamflow 2080s': slice('2069-10-01', '2099-09-30')}
for site in streamflow_sites['Site ID']:
    streamflow_file = '/Users/orianachegwidden/Downloads/CCSM4_RCP85-streamflow-1.0/'+                    'streamflow/CCSM4_RCP85_MACA_VIC_P2-'+site+'-streamflow-1.0.csv'
    df = get_model_ts(streamflow_file)
    for (label, timeframe) in streamflow_timeframes.items():
        collated_dataset.set_value(site, label, 
                                   metric_min7day_streamflow(df, 
                                        timeframe).quantile(q=0.1))

collated_dataset.to_csv('./sites_streamflow_stream_temperature.csv')

