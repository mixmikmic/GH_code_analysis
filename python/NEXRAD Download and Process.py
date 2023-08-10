import numpy as np
from numpy import ma
from pyproj import Geod
from bs4 import BeautifulSoup
import requests
from metpy.io.nexrad import Level3File
from metpy.plots import ctables
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os, tarfile, wget, re

tarfile_list = requests.get('http://www1.ncdc.noaa.gov/pub/has/HAS010805581/')
tar_list_soup = BeautifulSoup(tarfile_list.content, 'html.parser')
filename_list = tar_list_soup.select('td a')
noaa_filenames = list()

for filename in filename_list:
    noaa_filenames.append(filename['href'])

# Removing first because on the site will always be parent dir link
noaa_filenames = noaa_filenames[1:]
print(len(noaa_filenames))
print(noaa_filenames[:5])

def extract_data(file_obj):
    # Pull the data out of the file object
    f = Level3File(file_obj)
    datadict = f.sym_block[0][0]
    if 'data' not in datadict.keys():
        return None, None, None
    
    data = ma.array(datadict['data'])
    data[data==0] = ma.masked

    az = np.array(datadict['start_az'] + [datadict['end_az'][-1]])
    rng = np.linspace(0, f.max_range, data.shape[-1] + 1)

    # Data from MetPy needs to be converted to latitude and longitude coordinates
    g = Geod(ellps='clrk66')
    center_lat = np.ones([len(az),len(rng)])*f.lat
    center_lon = np.ones([len(az),len(rng)])*f.lon

    az2D = np.ones_like(center_lat)*az[:,None]
    rng2D = np.ones_like(center_lat)*np.transpose(rng[:,None])*1000
    lon,lat,back = g.fwd(center_lon,center_lat,az2D,rng2D)
    
    return lon, lat, data

def unstack_data(lon, lat, data):
    lat_df = pd.DataFrame(lat)
    lon_df = pd.DataFrame(lon)
    data_df = pd.DataFrame(data)
    
    lon_stack = lon_df.stack().reset_index()
    lon_stack = lon_stack.rename(columns={'level_0': 'x', 'level_1': 'y', 0: 'lon'})
    lat_stack = lat_df.stack().reset_index()
    lat_stack = lat_stack.rename(columns={'level_0': 'x', 'level_1': 'y', 0: 'lat'})
    coord_merge = pd.merge(lat_stack, lon_stack, on=['x', 'y'])
    
    data_stack = data_df.stack().reset_index()
    data_stack = data_stack.rename(columns={'level_0': 'x', 'level_1': 'y', 0: 'precip'})
    merged_data = pd.merge(coord_merge, data_stack, on=['x', 'y'], how='left')[['lat','lon','precip']]
    noaa_df = merged_data.dropna()
    return noaa_df
    
def spatial_join(noaa_df, gpd_file, group_col, file_time):
    #chi_wards = gpd.read_file('data/chicago_wards.geojson')
    geo_df = gpd.read_file(gpd_file)
    crs = {'init':'epsg:4326'}
    geometry = [Point(xy) for xy in zip(daa_df.lon, daa_df.lat)]
    noaa_geo = gpd.GeoDataFrame(noaa_df, crs=crs, geometry=geometry)
    merged_noaa = gpd.tools.sjoin(noaa_geo, geo_df, how='right', op='within').reset_index()
    
    noaa_grouped = merged_noaa.groupby([group_col])['precip'].mean().reset_index()
    noaa_grouped[group_col] = noaa_by_ward[group_col].astype(int)
    noaa_grouped.fillna(value=0, inplace=True)
    noaa_grouped.sort_values(by=group_col, inplace=True)
    noaa_grouped.to_csv('data/noaa_processed/{}_{}.csv'.format(group_col, file_time), index=False)

noaa_files = list()

tar = tarfile.open('nexrad_data/NWS_NEXRAD_NXL3_KLOT_20160830000000_20160830235959.tar.gz','r:gz')
for t in tar.getnames():
    if re.match(r'.*N1PLOT.*', t):
        noaa_files.append(t)
print(noaa_files[:5])
rand_noaa = tar.extractfile(noaa_files[4])
file_time = noaa_files[4].split('_')[-1]
lon, lat, data = extract_data(rand_noaa)

if lon is not None:
    processed_df = unstack_data(lon, lat, data)
    spatial_join(processed_df, 'data/chicago_wards.geojson', 'ward', file_time)
else:
    print('returned None')

# Using wget module to make long-running downloads easier
#file_path = wget.download('http://www1.ncdc.noaa.gov/pub/has/HAS010805581/NWS_NEXRAD_NXL3_KLOT_20160901000000_20160901235959.tar.gz',
#                          out='nexrad_data')

# We don't need most of the files in the tar file, use the tarfile module to only pull ones necessary
# 
# Once DAA files are downloaded, delete the rest of the archive
daa_files = list()

tar = tarfile.open('nexrad_data/NWS_NEXRAD_NXL3_KLOT_20160901000000_20160901235959.tar.gz',"r:gz")
for t in tar.getnames():
    if re.match(r'.*DAALOT.*', t):
        daa_files.append(t)
        
print(daa_files[20:30])
print(daa_files[29])

# Random DAA file that happens to have data
rand_daa = tar.extractfile(daa_files[29])
f = Level3File(rand_daa)

# Pull the data out of the file object
datadict = f.sym_block[0][0]
data = ma.array(datadict['data'])
data[data==0] = ma.masked

az = np.array(datadict['start_az'] + [datadict['end_az'][-1]])
rng = np.linspace(0, f.max_range, data.shape[-1] + 1)

# Data from MetPy needs to be converted to latitude and longitude coordinates
g = Geod(ellps='clrk66')
center_lat = np.ones([len(az),len(rng)])*f.lat
center_lon = np.ones([len(az),len(rng)])*f.lon

az2D = np.ones_like(center_lat)*az[:,None]
rng2D = np.ones_like(center_lat)*np.transpose(rng[:,None])*1000
lon,lat,back = g.fwd(center_lon,center_lat,az2D,rng2D)

# Once the data is returned, it can be converted into DataFrames for easier manipulation
lat_df = pd.DataFrame(lat)
lon_df = pd.DataFrame(lon)
data_df = pd.DataFrame(data)
print(lat_df.shape)
lat_df.head()

# Stack DataFrames so dealing with more rows than columns
lon_stack = lon_df.stack().reset_index()
lon_stack = lon_stack.rename(columns={'level_0': 'x', 'level_1': 'y', 0: 'lon'})
lat_stack = lat_df.stack().reset_index()
lat_stack = lat_stack.rename(columns={'level_0': 'x', 'level_1': 'y', 0: 'lat'})
print(lon_stack.shape)
lon_stack.head()

# Merge lat and lon DataFrames on x and y indices from original matrix to join data_df
coord_merge = pd.merge(lat_stack, lon_stack, on=['x', 'y'])
print(coord_merge.shape)
coord_merge.head()

# Do the same with the precipitation DataFrame, and then merge it back onto 
data_stack = data_df.stack().reset_index()
data_stack = data_stack.rename(columns={'level_0': 'x', 'level_1': 'y', 0: 'precip'})
print(data_stack.shape)
data_stack.head()

# This file had more information for precipitation, so the values are all on a scale of 0 to 255
data_stack['precip'].unique()

# Merge coordinates and precipitation data together
merged_data = pd.merge(coord_merge, data_stack, on=['x', 'y'], how='left')
print(merged_data.shape)
merged_data.head()

# To reduce data size and ignore unnecessary data, we can drop the many NaN rows
daa_df = merged_data.dropna()
print(daa_df.shape)
daa_df.head()

chi_wards = gpd.read_file('data/chicago_wards.geojson')
print(type(chi_wards))
chi_wards.head()

# Convert lat and lon into shapely Point objects and make into GeoDataFrame
# Important that the crs values are the same
crs = {'init':'epsg:4326'}
geometry = [Point(xy) for xy in zip(daa_df.lon, daa_df.lat)]
daa_geo = gpd.GeoDataFrame(daa_df, crs=crs, geometry=geometry)
daa_geo.head()

# Spatial join, important for speed that op is 'within', and retain all boundary keys with right join
ward_daa = gpd.tools.sjoin(daa_geo, chi_wards, how='right', op='within')
ward_daa.head()

ward_daa_df = ward_daa.reset_index()
ward_daa_df.head()

daa_by_ward = ward_daa_df.groupby(['ward'])['precip'].mean().reset_index()
daa_by_ward['ward'] = daa_by_ward['ward'].astype(int)
daa_by_ward.fillna(value=0, inplace=True)
daa_by_ward.sort_values(by='ward', inplace=True)
daa_by_ward.head()

chi_wards['ward'] = chi_wards['ward'].astype(int)
chi_ward_precip = chi_wards.merge(daa_by_ward, on='ward')
print(chi_ward_precip.dtypes)
print(chi_ward_precip.shape)
chi_ward_precip.head()

# Save to inches with units as millimeters
chi_ward_precip.to_file('data/chi_ward_precip_millim.geojson', driver='GeoJSON')

# Converting precip to inches, saving to file
chi_ward_in = chi_ward_precip.copy()
chi_ward_in['precip'] = chi_ward_in['precip'].apply(lambda x: x / 25.4)
chi_ward_in.to_file('data/chi_ward_precip_in.geojson', driver='GeoJSON')

# When doing actual ETL for multiple files, will want to delete tar archives, keeping for now
# os.remove(file_path)

