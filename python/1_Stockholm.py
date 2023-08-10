import pandas as pd
import numpy as np
import geopandas as gpd
from geopandas import GeoDataFrame
import fiona
from fiona.crs import from_epsg
import shapely
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf
from sklearn import cluster, datasets
import matplotlib.pylab as plt
#from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
get_ipython().magic('matplotlib inline')

get_ipython().magic('matplotlib inline')

# Dataset_1: Importing the Streetline Datafile

nyc_streetline = gpd.read_file('Data/Streetline/geo_export_bd3fc424-f8ad-4c3b-9d41-362af6349e2e.shp')
nyc_streetline.head()

# Dataset_1: Changing the Co-ordinate system for Uniformity
nyc_streetline.crs = from_epsg(4326)
nyc_streetline.to_crs(epsg=4326, inplace=True)

# Dataset_1 : Filtering and Cleaning Unnecessary Data
nyc_streetline = nyc_streetline[[ u'physicalid', u'geometry', u'st_width', u'shape_leng', u'bike_lane', 
                                 u'borocode', u'l_zip', u'rw_type' ]]

mn_streetline = nyc_streetline[nyc_streetline.borocode == '1']
mn_streetline = mn_streetline[mn_streetline.rw_type == 1]
mn_streetline = mn_streetline[mn_streetline.st_width != 0]

#Resetting Index after clean
mn_streetline.reset_index(inplace=True)
mn_streetline.drop('index', axis=1, inplace=True)

#Renaming Columns after reset
mn_streetline.columns = ['physicalid', 'geometry', 'Street_width_1', 'Shape_length_1', 'bike_lane_1',  u'borocode',                         u'l_zip', u'rw_type']
mn_streetline.drop(['borocode', 'l_zip', 'rw_type'],axis=1, inplace=True)
mn_streetline.head()

# Dataset_1 : Truncated Version for running the code
x_streetline = mn_streetline.loc[:100,:]
x_streetline.head()

# Dataset_1 : Sample Plotting of Truncated Data
x_streetline.plot(figsize=(10,10))
plt.show()

# Dataset_2: Importing the Potholes Shape File Data (Sourced from Open Data - refered by Varun)

nyc_pthls = gpd.read_file("Data/closed-potholes-datafeeds/ClosedPotholes_201608161516.shp")
nyc_pthls.head()

# Dataset_2: Changing the Co-ordinate system for Uniformity
nyc_pthls.crs = from_epsg(2263)
nyc_pthls.to_crs(epsg=4326, inplace=True)
nyc_pthls.head()

# Dataset_2: Filtering and Cleaning Unnecessary Data

# Truncating the initial datset to consider only Manhattan
mn_pthls = nyc_pthls[nyc_pthls['Boro'] == 'M']
mn_pthls.reset_index(inplace=True)
mn_pthls.drop('index', axis=1, inplace=True)
mn_pthls = gpd.GeoDataFrame(mn_pthls[['geometry','DefNum']])
mn_pthls.head()

mn_pthls.crs = from_epsg(4326)
mn_pthls.crs

# Dataset_2 : Truncated Version for running the code

x_pthls = mn_pthls.loc[:100,:]
x_pthls.head()

# Dataset_2: Trying to group the No: of Potholes per Linestring

x_pthls_count = gpd.GeoDataFrame(x_pthls.geometry.value_counts())
x_pthls_count.reset_index(inplace=True)
x_pthls_count.columns = ['geometry','Potholes']
x_pthls_count.head()

# Dataset_2: Changing the Co-ordinate system for Uniformity
x_pthls_count.crs = from_epsg(4326)
x_pthls_count.head()



# Dataset_3 : Traffic Volume (Sourced from Open Data - AD)

# Cleaned in Local System due to compute errors and uploaded clean data here
nyc_traf_vol = gpd.read_file('Data/Traffic_Volume/Traffic_Volume.shp')
nyc_traf_vol.head()

# Dataset_3 : Rearranging the dataset and Filtering for convenience

nyc_traf_vol = nyc_traf_vol[['geometry', 'AADT', 'Shape_Leng']]
nyc_traf_vol.columns = ['geometry', 'Traf_Vol_3', 'Shape_Length_3']
nyc_traf_vol.head()

# Dataset_3 :  Changing the Co-ordinate system for Uniformity
nyc_traf_vol.crs = from_epsg(4326)
nyc_traf_vol.head()

# Dataset_4 : MTA Bus Stops (Sourced from Open Data - VI)

nyc_busstops = gpd.read_file('Data/Bus_Shapefiles/BusStopsAsOfMarch2.shp')
nyc_busstops.head()

# Dataset_4 : Filtering and Cleaning Unnecessary Data

nyc_busstops = nyc_busstops[['geometry','box_id']]
nyc_busstops.head()

#Dataset_4 : Checking the Coordinate System
nyc_busstops.crs



# Dataset_5: Zip Code Level Data (Sourced from Open Data - VI)

nyc_zips = gpd.read_file('Data/ZIP_CODE_040114/ZIP_CODE_040114.shp')
nyc_zips.head()

# Dataset_5 : Filtering and Cleaning Unnecessary Data
nyc_zips = nyc_zips[['geometry', 'ZIPCODE']]
nyc_zips.head()

# Dataset_5 :  Changing the Co-ordinate system for Uniformity
nyc_zips.crs = from_epsg(2263)
nyc_zips.to_crs(epsg=4326, inplace=True)
nyc_zips.head()

nyc_zips.crs



# 1_Streetline Data: Spatial Join for Zipcodes Data
mn_streetline_zip = gpd.sjoin(mn_streetline, nyc_zips, how="inner", op='intersects')
mn_streetline_zip.drop('index_right',axis=1, inplace=True)
mn_streetline_zip.head()

# 1_Streetline Data for 10036 Zip Code

z10036_streetline = mn_streetline_zip[mn_streetline_zip['ZIPCODE'] == '10036']
z10036_streetline.reset_index(inplace=True)
z10036_streetline.drop('index', axis=1, inplace=True)
z10036_streetline.head()

# z10036_Dataset_1 : Sample Plotting of Truncated Data
z10036_streetline.plot(figsize=(5,5))
plt.show()

# 2_Potholes Data: Spatial Join for Zipcodes Data
mn_pthls_zip = gpd.sjoin(mn_pthls, nyc_zips, how="inner", op='intersects')
mn_pthls_zip.drop('index_right',axis=1, inplace=True)
mn_pthls_zip.head()

# 2_Potholes Data for 10036 Zip Code

z10036_pthls = mn_pthls_zip[mn_pthls_zip['ZIPCODE'] == '10036']
z10036_pthls.reset_index(inplace=True)
z10036_pthls.drop('index', axis=1, inplace=True)
z10036_pthls.head()

# z10036_Dataset_2 : Sample Plotting of Truncated Data
z10036_pthls.plot(figsize=(5,5))
plt.show()

z10036_pthls_count = gpd.GeoDataFrame(z10036_pthls.geometry.value_counts())
z10036_pthls_count.reset_index(inplace=True)
z10036_pthls_count.columns = ['geometry','Potholes']
z10036_pthls_count.head()

# z10036_Dataset_2 : Sample Plotting of Truncated Data
z10036_pthls.plot(figsize=(5,5))
plt.show()

# 3_Traffic Volume Data: Spatial Join for Zipcodes Data
nyc_traf_vol_zip = gpd.sjoin(nyc_traf_vol, nyc_zips, how="inner", op='intersects')
nyc_traf_vol_zip.drop('index_right',axis=1, inplace=True)
nyc_traf_vol_zip.head()

# 3_Traffic Volume Data for 10036 Zip Code

z10036_traf_vol = nyc_traf_vol_zip[nyc_traf_vol_zip['ZIPCODE'] == '10036']
z10036_traf_vol.reset_index(inplace=True)
z10036_traf_vol.drop('index', axis=1, inplace=True)
z10036_traf_vol.head()

# z10036_Dataset_3 : Sample Plotting of Truncated Data
z10036_traf_vol.plot(figsize=(5,5))
plt.show()

# 4_Bus Stops Data: Spatial Join for Zipcodes Data
nyc_busstops_zip = gpd.sjoin(nyc_busstops, nyc_zips, how="inner", op='intersects')
nyc_busstops_zip.drop('index_right',axis=1, inplace=True)
nyc_busstops_zip.head()

nyc_busstops.crs = nyc_zips.crs

nyc_zips.crs

nyc_busstops.crs

# 4_Bus Stops Data for 10036 Zip Code

z10036_busstops = nyc_busstops_zip[nyc_busstops_zip['ZIPCODE'] == '10036']
z10036_busstops.reset_index(inplace=True)
z10036_busstops.drop('index', axis=1, inplace=True)
z10036_busstops.head()

# z10036_Dataset_4 : Sample Plotting of Truncated Data
z10036_busstops.plot(figsize=(5,5))
plt.show()

z10036_pthls_count.crs = z10036_streetline.crs

z10036_1 = gpd.sjoin(z10036_streetline,z10036_traf_vol, how="inner", op='intersects')
len(z10036_1)

z10036_1.columns

z10036_1 = z10036_1[[ u'geometry',u'Traf_Vol_3', u'Street_width_1', u'Shape_length_1', u'Shape_Length_3']]
z10036_1.sort_index(inplace=True)
z10036_1.reset_index(inplace=True)
z10036_1.drop('index', axis=1, inplace=True)
z10036_1.head()

#Using Normal Standardization
reg_data = pd.DataFrame()

reg_data['Traf_Vol_3'] = (z10036_1['Traf_Vol_3']-z10036_1['Traf_Vol_3'].mean())/z10036_1['Traf_Vol_3'].std()
reg_data['Street_width_1'] = (z10036_1['Street_width_1']-z10036_1['Street_width_1'].mean())/z10036_1['Street_width_1'].std()
reg_data['Shape_length_1'] = (z10036_1['Shape_length_1']-z10036_1['Shape_length_1'].mean())/z10036_1['Shape_length_1'].std()

reg_data.head()

#Feature Scaling
reg_data_1 = pd.DataFrame()
reg_data_1['Traf_Vol_3'] = (z10036_1['Traf_Vol_3']-z10036_1['Traf_Vol_3'].min())/(z10036_1['Traf_Vol_3'].max()-z10036_1['Traf_Vol_3'].min())
reg_data_1['Street_width_1'] = (z10036_1['Street_width_1']-z10036_1['Street_width_1'].min())/(z10036_1['Street_width_1'].max()-z10036_1['Street_width_1'].min())
reg_data_1['Shape_length_1'] = (z10036_1['Shape_length_1']-z10036_1['Shape_length_1'].min())/(z10036_1['Shape_length_1'].max()-z10036_1['Shape_length_1'].min())

reg_data_1.head()

lm = smf.ols(formula = 'Traf_Vol_3 ~ Street_width_1 + Shape_length_1', data = reg_data).fit()
print lm.summary()
#print(lm.params[1:])
#print('R^2', lm.rsquared)

lm = smf.ols(formula = 'Traf_Vol_3 ~ Street_width_1 + Shape_length_1', data = reg_data_1).fit()
print lm.summary()



