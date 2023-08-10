get_ipython().magic('matplotlib inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# OpenDAP URLs for each product
ndbc_rt_url='http://dods.ndbc.noaa.gov/thredds/dodsC/data/swden/41047/41047w9999.nc'
ndbc_2014_url = 'http://dods.ndbc.noaa.gov/thredds/dodsC/data/swden/41047/41047w2014.nc'
planetos_tds_url = 'http://thredds.planetos.com/thredds/dodsC/dpipe//rel_0_8x03_dataset/transform/ns=/noaa_ndbc_swden_stations/scheme=/http/authority=/dods.ndbc.noaa.gov/path=/thredds/dodsC/data/swden/41047/41047w9999.nc/chunk=/1/1/data'

# acquire OpenDAP datasets
ds_ndbc_rt = xr.open_dataset(ndbc_rt_url)
ds_ndbc_2014 = xr.open_dataset(ndbc_2014_url)
ds_planetos = xr.open_dataset(planetos_tds_url)

# Let's focus on a specific hour of interest...
time = '2014-08-09 00:00:00'

# Select the specific hour for each dataset
ds_ndbc_rt_hour = ds_ndbc_rt.sel(time=time).isel(latitude=0, longitude=0)
ds_ndbc_2014_hour = ds_ndbc_2014.sel(time=time).isel(latitude=0, longitude=0)
ds_planetos_hour = ds_planetos.sel(time=time).isel(latitude=0, longitude=0)

# First, the Planet OS data which is acquired from the NDBC realtime station file.
df_planetos = ds_planetos_hour.to_dataframe().drop(['context_time_latitude_longitude_frequency','mx_dataset','mx_creator_institution'], axis=1)
df_planetos.head(8)

# Second, the NDBC realtime station data.
df_ndbc_rt = ds_ndbc_rt_hour.to_dataframe()
df_ndbc_rt.head(8)

# Finally, the 2014 archival data.
df_ndbc_2014 = ds_ndbc_2014_hour.to_dataframe()
df_ndbc_2014.head(8)

df_planetos.describe()

df_ndbc_rt.describe()

df_ndbc_2014.describe()

# function below requires identical index structure
def df_diff(df1, df2):
    ne_stacked = (df1 != df2).stack()
    changed = ne_stacked[ne_stacked]
    difference_locations = np.where(df1 != df2)
    changed_from = df1.values[difference_locations]
    changed_to = df2.values[difference_locations]
    return pd.DataFrame({'df1': changed_from, 'df2': changed_to}, index=changed.index)

# Compare the NDBC realtime to Planet OS data
# Note that NaN != NaN evaluates as True, so NaN values will be raised as inconsistent across the dataframes
# We could use fillna() to fix this issue, however this is not implemented here.
df_diff(df_ndbc_rt, df_planetos)

plt.figure(figsize=(20,10))
ds_ndbc_rt_hour.spectral_wave_density.plot(label='NDBC Realtime')
ds_ndbc_2014_hour.spectral_wave_density.plot(label='NDBC 2014')
ds_planetos_hour.spectral_wave_density.plot(label='Planet OS')
plt.legend()
plt.show()

vars = ['wave_spectrum_r1','wave_spectrum_r2']
df_planetos.loc[:,vars].plot(label="Planet OS", figsize=(18,6))
df_ndbc_rt.loc[:,vars].plot(label="NDBC Realtime", figsize=(18,6))
df_ndbc_2014.loc[:,vars].plot(label="NDBC 2014", figsize=(18,6))
plt.show()

vars = ['principal_wave_dir','mean_wave_dir']
df_planetos.loc[:,vars].plot(label="Planet OS", figsize=(18,6))
df_ndbc_rt.loc[:,vars].plot(label="NDBC Realtime", figsize=(18,6))
df_ndbc_2014.loc[:,vars].plot(label="NDBC 2014", figsize=(18,6))
plt.show()



