import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm  # colormaps for plottting

get_ipython().magic('matplotlib inline')

# open netcdf file
data_url = 'http://hydromet-thredds.princeton.edu:9000/thredds/dodsC/MonitoringStations/broadmead.nc'
ds = xr.open_dataset(data_url)

# select 4 components of radiation
radiation_ds = ds[['Rl_downwell_Avg', 'Rl_upwell_Avg', 'Rs_downwell_Avg', 'Rs_upwell_Avg']]

# convert to a pandas.dataframe object
radiation_df = radiation_ds.to_dataframe()

# set up the framework for the subplots. In this case the figure will 
# have a width of 14 and a height of 4. In this figure there will be 
# two sets of axes arranged in 2 columns and 1 row
fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(14,4))

# axes now contains a list of the subplots. To refer to the first of the
# subplots we call the first item from the list: axes[0]
radiation_df.boxplot(column='Rl_upwell_Avg', by=radiation_df.index.month, 
                     whis= [5, 95], sym='', ax=axes[0])
axes[0].set_xlabel('months') # set the xlabel on the first subplot
axes[0].set_title('Upwelling longwave radiation by month')
axes[0].set_ylim(200, 550) # set the y limits on the first subplot

radiation_df.boxplot(column='Rl_downwell_Avg', by=radiation_df.index.month, 
                     whis= [5, 95], sym='', ax=axes[1])
axes[1].set_xlabel('months')
axes[1].set_title('Downwelling longwave radiation by month')
axes[1].set_ylim(200, 550)

plt.suptitle('') # this makes the super title for the whole figure blank
plt.show()

# pivot the data to make the index the day of year and the columns the years
pv = pd.pivot_table(radiation_df, 
                    index=radiation_df.index.dayofyear, 
                    columns=radiation_df.index.year, 
                    values='Rl_upwell_Avg', aggfunc='mean')

# plot the aggregated data, since we don't really care which year is which
# we can turn the legend off and set all the colors to gray
pv.plot(figsize=(12,5), legend=False, color='gray')

# plot the data that has been averaged across years
pv.mean(axis=1).plot(linewidth=2, color='red')
plt.ylabel('Upwelling longwave radiation [W/m^2]')
plt.xlabel('days of year')
plt.title('Annual cycle of upwelling longwave radiation over period of record')
plt.show()

# by changing just these two variables, you can create a whole new presentation-quality plot
var = 'Rs_downwell_Avg'
title = 'Monthly diurnal cycle for downwelling shortwave radiation'

# pivot the data to make the index time of day and the columns the months
pv = pd.pivot_table(radiation_df,
                    index=radiation_df.index.time, 
                    columns=radiation_df.index.month, 
                    values=var)

# set the colors to a discretized circular colormap to fit cyclical data
pv.plot(figsize=(12,5), color=cm.hsv(np.linspace(0, 1, 12)), linewidth=2, title=title)
plt.legend(title='Months', loc='upper left')

# using this method we can create labels that depend only on var
plt.ylabel('{var} [{units}]'.format(var=var, units=ds[var].units))

plt.savefig(title+'.png')
plt.show()

df = None
for v in ['Rl_downwell_Avg', 'Rl_upwell_Avg', 'Rs_downwell_Avg', 'Rs_upwell_Avg']:
    pv = pd.pivot_table(radiation_df, 
                        index=radiation_df.index.time, 
                        columns=radiation_df.index.month, 
                        values=v)
    df_rad = pd.DataFrame(pv.mean(axis=1), columns=[v])
    df = pd.concat([df, df_rad], axis=1)
df.plot(figsize=(12,5), linewidth=2)
plt.show()

data_url = 'http://hydromet-thredds.princeton.edu:9000/thredds/dodsC/MonitoringStations/butler.nc'
ds = xr.open_dataset(data_url)

# convert to a pandas.dataframe object
df = ds['Rain_mm_3_Tot'].to_dataframe()

# pivot the data to make the index the accumulated rainfall and the columns the years. 
pv = pd.pivot_table(df, index=df.index.month, columns=df.index.year, values='Rain_mm_3_Tot', aggfunc='sum')

# take the mean across the columns (years) and then plot it as a bar graph
pv.mean(axis=1).plot.bar(figsize=(12,5))
plt.ylabel('Accumulated Rainfall (mm)')
plt.xlabel('month')
plt.title('Monthly Accumulated Rainfall across period of record')
plt.show()

data_url = 'http://hydromet-thredds.princeton.edu:9000/thredds/dodsC/MonitoringStations/butler.nc'
ds = xr.open_dataset(data_url)

# resample to a daily time scale by summing across time (other units are 'min', 'H')
ds_rain = ds['Rain_mm_3_Tot'].resample('1D','time', how='sum', label='right')

# convert to a pandas.Series object and then sort so that the largest values are at the top
wettest = ds_rain.to_pandas().sort_values(ascending=False)

# print the top 5 values
wettest.head(5)

data_url = 'http://hydromet-thredds.princeton.edu:9000/thredds/dodsC/MonitoringStations/butler.nc'
ds = xr.open_dataset(data_url)

# use the where command to filter data as it comes in and drop all the times at which the criteria aren't met
rain_on_wet_soil = ds.where((ds['Rain_mm_3_Tot']>0) & (ds['VW']>.30)).dropna('time')

# check out what these times look like
rain_on_wet_soil.to_dataframe()[['Rain_mm_3_Tot', 'VW']].describe()

data_url = 'http://hydromet-thredds.princeton.edu:9000/thredds/dodsC/MonitoringStations/broadmead_parsivel.nc'
ds = xr.open_dataset(data_url)['rain_intensity']

# resample to match time step and divide by 60 to convert from rate to accumulation/
df_parsivel = ds.to_pandas().resample('1min', label='right').mean()/60
df_parsivel.name = 'parsivel'

data_url = 'http://hydromet-thredds.princeton.edu:9000/thredds/dodsC/MonitoringStations/broadmead_1min.nc'
ds= xr.open_dataset(data_url)[['Rain_1_mm_Tot', 'Rain_2_mm_Tot']]
df_tipping = ds.to_dataframe()[['Rain_1_mm_Tot', 'Rain_2_mm_Tot']]

# join the datasets and only keep the time steps when data is availble for every variable (how='inner')
df = df_tipping.join(df_parsivel, how='inner')

# change the names of the columns to something more descriptive
df.columns = ['tipping_1', 'tipping_2', 'parsivel_1min']

fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(14,6))
df.plot.scatter('tipping_1', 'parsivel_1min', ax=axes[0])
df.plot.scatter('tipping_2', 'parsivel_1min', ax=axes[1])
plt.show()

df_hourly = df.resample('1H', label='right').sum()
fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(14,6))
df_hourly.plot.scatter('tipping_1', 'parsivel_1min', ax=axes[0])
df_hourly.plot.scatter('tipping_2', 'parsivel_1min', ax=axes[1])
plt.show()

