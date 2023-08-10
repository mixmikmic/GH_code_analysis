get_ipython().run_line_magic('matplotlib', 'inline')
# above generates plots in line within this page

import pandas as pd # pandas module
import numpy as np # numpy module
import netCDF4 as nc # netcdf module
import matplotlib.pyplot as plt # plot from matplotlib module

in_nc = nc.Dataset("../data/indata/soil_moist_20min_Kendall_AZ_n1400.nc") # read file
print(in_nc) # print file information

y = in_nc.variables['lat'][:] # read latitutde variable
x = in_nc.variables['lon'][:] # read longitude variable
print("Latitude: %.5f, Longitude: %.5f" % (y,x)) # print latitutde, longitude

soil_moisture = in_nc.variables['soil_moisture'][:] # read soil moisture variable
print(in_nc.variables['soil_moisture']) # print the variable attributes

depth = in_nc.variables['depth'][:] # read depth variable
print(in_nc.variables['depth']) # print the variable attributes

time = in_nc.variables['time'][:] # read time variable
print(in_nc.variables['time']) # print the variable attributes

time_unit = in_nc.variables["time"].getncattr('units') # first read the 'units' attributes from the variable time
time_cal = in_nc.variables["time"].getncattr('calendar') # read calendar type
local_time = nc.num2date(time, units=time_unit, calendar=time_cal) # convert time
print("Original time %s is now converted as %s" % (time[0], local_time[0])) # check conversion

sm_df = pd.DataFrame(soil_moisture, columns=depth, index=local_time.tolist()) # read into pandas dataframe
print(sm_df[:5]) # print the first 5 rows of dataframe

sm_df_daily = sm_df.groupby(pd.TimeGrouper('1D')).aggregate(np.nanmean) # convert to daily.
print(sm_df_daily[:5]) # print the first 5 rows

ylabel_name = in_nc.variables["soil_moisture"].getncattr('long_name') + ' (' +               in_nc.variables["soil_moisture"].getncattr('units') + ')' # Label for y-axis
series_name = in_nc.variables["depth"].getncattr('long_name') + ' (' +               in_nc.variables["depth"].getncattr('units') + ')' # Legend title
# plot
plt.figure()
sm_df_daily.plot()
plt.legend(title=series_name)
plt.ylabel(ylabel_name)

sm_df_daily.to_csv("../data/outdata/daily_soilscape.csv", index_label="DateTime") # Daily
sm_df.to_csv("../data/outdata/original_soilscape.csv", index_label="DateTime") # Original

in_nc.close()

