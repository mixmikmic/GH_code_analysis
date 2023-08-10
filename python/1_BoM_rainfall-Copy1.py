import datacube
import xarray as xr
import pandas as pd
get_ipython().magic('matplotlib inline')
from datacube.storage.masking import mask_invalid_data
from datacube.storage.storage import write_dataset_to_netcdf

#app is a note to let GA know what we are doing with it, user-defined
#when loading data, #make sure data is on same coordinate scale or convert 
dc=datacube.Datacube(app='learn-data-access')
dc

products = dc.list_products()
#products.columns.tolist()

display_columns = ['name', 'description', 'platform', 'product_type', 'instrument', 'crs', 'resolution']
#display_columns

# #list only nbar products
Rainfall_list = products[products['product_type'] == 'rainfall'][display_columns].dropna()
# productlist = products[display_columns].dropna()
Rainfall_list.describe

query = {
    'time': ('1960-01-01', '2013-12-31'),
    'lat': (-24.5, -38), # Bounding Lat
    'lon': (138.5, 152.5), # Bongding Long
}
#query = {
#    'time': ('2000-01-01', '2013-12-31'),
#    'geopolygon': geom,
# }

# attempt BoM rainfall 
#2 stars unpack the limits of our query, we load specific measurements from a product
Rainfall_data_long = dc.load(product='bom_rainfall_grids', measurements=['rainfall'], **query)

Rainfall_data_long

Rainfalldata = mask_invalid_data(Rainfall_data_long)
Rainfalldata.nbytes/10**9

from datacube.storage.storage import write_dataset_to_netcdf
# save daily rainfall to netcdf file 
path = '/g/data/oe9/project/team-drip/Rainfall/daily_rainfall_MDB_1960_2013.nc'
# Rainfalldata.to_netcdf(path)
write_dataset_to_netcdf(Rainfalldata,path)

#path = '/home/563/sl1412/rainfall/rainfallMDB2.nc'
# Rainfalldata.to_netcdf(path)

# Rainfalldata.assign_attrss(4326)

Rainfalldata.rainfall.isel(time=0).plot.imshow() # selection of time = 0 for all MDB

# calculate monthly rainfall 
Rainmonth=Rainfalldata.resample(time="1M").sum()

Rainfalldata

attr = Rainfalldata.attrs
attr

Rainmonth.attrs = attr

Rainmonth

Rainmonth.rainfall.isel(time=1).plot.imshow() # selection of time = 1



