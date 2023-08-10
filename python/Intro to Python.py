# load some key packages:
import numpy as np                # a numerical package
import pandas as pd               # a data analysis package
import matplotlib.pyplot as plt   # a scientific plotting package

# to display the plots in the same document
get_ipython().magic('matplotlib inline')

# define an array of points (start, end, by)
x = np.arange(0,6*np.pi,0.1)

# check the array
print(x)

# plot the sin of these points
plt.plot(x, np.sin(x), 'blue')

# plot the cos of these points
plt.plot(x, np.cos(x),'red')

# show that plot below
plt.show()

# load the netcdf-handling package:
import xarray as xr

# from THREDDS server. 
data_url = 'http://hydromet-thredds.princeton.edu:9000/thredds/dodsC/MonitoringStations/butler.nc'

# open the file and assign it the name: ds
ds = xr.open_dataset(data_url)

# check it out
print(ds)

variable = 'AirTC_Avg'

# check it out
print(ds[variable])

# plot all the data
ds[variable].plot()

# convert to a pandas object and then to a dataframe
df = ds[variable].to_pandas().to_frame(name=variable)

# get a summary of the data including the percentiles listed
df.describe(percentiles=[.1,.25,.5,.75,.9])

df['2012-06-03'].boxplot(column=variable, by=df['2012-06-03'].index.hour)
# set the labels
plt.xlabel(' ')
plt.ylabel('Temperature [C]')
plt.title('Monthly boxplots')
plt.suptitle('')

plt.show()

# create a box plot
df.boxplot(column=variable, by=df.index.month, whis= [10, 90], sym='')

# set the labels
plt.xlabel('month')
plt.ylabel('Temperature [C]')
plt.title('Monthly boxplots')
plt.suptitle('')

plt.show()

# choose a date period (such as a month)
a_month = ds[variable].sel(time='2016-01')

# or grab the range between two specific days
a_week =  ds[variable].sel(time=slice('2015-07-06', '2015-07-13'))

# Create a figure with two subplots 
fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(14,4))

# plot the month of data in the first subplot
a_month.plot(ax=axes[0])
axes[0].set_title('A month')

# plot the week of data in the first subplot
a_week.plot(ax=axes[1])
axes[1].set_title('A week')

plt.show()

# slice the dataset by time and grab variables of interest
vars_for_a_week = ds[['Rain_mm_3_Tot', 'VW']].sel(time=slice('2015-07-06', '2015-07-13'))
print(vars_for_a_week)

# convert to pandas.dataframe
df = vars_for_a_week.to_dataframe()[['Rain_mm_3_Tot', 'VW']]
df.head()

# plot on left and right axes
df.plot(secondary_y='Rain_mm_3_Tot', figsize=(12,4))

# by setting the limits as (max, min) we flip the axis so that rain comes down from the top
plt.ylim(12,0)
plt.show()

data_url = 'http://hydromet-thredds.princeton.edu:9000/thredds/dodsC/MonitoringStations/broadmead.nc'
ds = xr.open_dataset(data_url)
broadmead_rain_ds = ds[['Rain_1_mm_Tot', 'Rain_2_mm_Tot']].sel(time=slice('2016-02-23', '2016-02-26'))
broadmead_rain = broadmead_rain_ds.to_dataframe().drop(['lat','lon','station_name'], axis=1)
ds.close()

data_url = 'http://hydromet-thredds.princeton.edu:9000/thredds/dodsC/MonitoringStations/washington_lake.nc'
ds = xr.open_dataset(data_url)
washington_lake_level_ds = ds['Lvl_cm_Avg'].sel(time=slice('2016-02-23', '2016-02-26'))
washington_lake_level = washington_lake_level_ds.to_dataframe().drop(['lat','lon','station_name'], axis=1)
ds.close()

data_url = 'http://hydromet-thredds.princeton.edu:9000/thredds/dodsC/MonitoringStations/washington_up.nc'
ds = xr.open_dataset(data_url)
washington_up_level_ds = ds['Corrected_cm_Avg'].sel(time=slice('2016-02-23', '2016-02-26'))
washington_up_level = washington_up_level_ds.to_dataframe().drop(['lat','lon','station_name'], axis=1)
ds.close()

data_url = 'http://hydromet-thredds.princeton.edu:9000/thredds/dodsC/MonitoringStations/washington_down.nc'
ds = xr.open_dataset(data_url)
washington_down_level_ds = ds['Corrected_cm_Avg'].sel(time=slice('2016-02-23', '2016-02-26'))
washington_down_level = washington_down_level_ds.to_dataframe().drop(['lat','lon','station_name'], axis=1)
ds.close()

washington_up_storm = washington_up_level-washington_up_level.iloc[0,0]
washington_down_storm = washington_down_level-washington_down_level.iloc[0,0]
washington_lake_storm = washington_lake_level-washington_lake_level.iloc[0,0]

# create a figure with 2 subplots 
fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(12,10), sharex=True)

broadmead_rain.plot(ax=axes[0], linewidth=2)
washington_up_storm.plot(ax=axes[1], linewidth=2)
washington_down_storm.plot(ax=axes[1], linewidth=2)
washington_lake_storm.plot(ax=axes[1], linewidth=2)

# set titles and legends
plt.suptitle('Timing of Rainfall and Stream Depth peak during February Storm', fontsize=18)
axes[0].set_title('Rainfall (mm)')
axes[0].set_ylabel('5 min rain (mm)')

axes[1].set_title('Stream level minus base level (cm)')
axes[1].legend(['upstream','downstream', 'lake'])
axes[1].set_ylabel('Storm depth (cm)')
axes[1].set_xlabel('Time in UTC')

# save fig to current folder
plt.savefig('Rain and discharge.png')
plt.show()

def select(site, var, start, end):
    """
    Select data from netcdf file hosted on the Princeton Hydrometeorology thredds server

    Parameters
    -----------
    site: one of the monitoring stations in quotes ('broadmead')
    var: one of the variables from this site in quotes ('Rain_1_mm_Tot'), 
         or a list of variables(['Hc', 'Hs'])
    start: starting time for data.frame ('YYYY-MM-DD hh:mm:ss')
    end: ending time for data.frame ('YYYY-MM-DD hh:mm:ss')

    Returns
    -------
    df: pandas.DataFrame object with time index and the variable(s) as the column(s)
    """

    import xarray as xr

    data_url = 'http://hydromet-thredds.princeton.edu:9000/thredds/dodsC/MonitoringStations/'+ site+'.nc'
    ds = xr.open_dataset(data_url)
    _ds = ds[var].sel(time=slice(start, end))
    df = _ds.to_dataframe().drop(['lat','lon','station_name'], axis=1)
    ds.close()
    return df

broadmead_rain = select('broadmead', ['Rain_1_mm_Tot','Rain_2_mm_Tot'], '2016-02-23', '2016-02-26 12:00')
washington_lake_level = select('washington_lake', 'Lvl_cm_Avg', '2016-02-23', '2016-02-26 12:00')
washington_down_level = select('washington_down', 'Corrected_cm_Avg', '2016-02-23', '2016-02-26 12:00:00')
washington_up_level = select('washington_up', 'Corrected_cm_Avg', '2016-02-23', '2016-02-26 12:00')

washington_up_storm = washington_up_level-washington_up_level.iloc[0,0]
washington_down_storm = washington_down_level-washington_down_level.iloc[0,0]
washington_lake_storm = washington_lake_level-washington_lake_level.iloc[0,0]

# create a figure with 2 subplots 
fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(12,10), sharex=True)

broadmead_rain.plot(ax=axes[0], linewidth=2)
washington_up_storm.plot(ax=axes[1], linewidth=2)
washington_down_storm.plot(ax=axes[1], linewidth=2)
washington_lake_storm.plot(ax=axes[1], linewidth=2)

# set titles and legends
plt.suptitle('Timing of Rainfall and Stream Depth peak during February Storm', fontsize=18)
axes[0].set_title('Rainfall (mm)')
axes[0].set_ylabel('5 min rain (mm)')

axes[1].set_title('Stream level minus base level (cm)')
axes[1].legend(['upstream','downstream', 'lake'])
axes[1].set_ylabel('Storm depth (cm)')
axes[1].set_xlabel('Time in UTC')

plt.show()

