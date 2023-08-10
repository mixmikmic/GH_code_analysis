from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import metpy.calc as mpcalc
from metpy.io import get_upper_air_data
from metpy.plots import SkewT
from metpy.units import units
import numpy as np
import pandas as pd

get_ipython().magic('matplotlib inline')

def get_sounding_parameters(ds):
    date = ds.name.to_pydatetime()
    station = 'OUN'
    try:
        dataset = get_upper_air_data(date, station)
    except:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    
    p = dataset.variables['pressure'][:]
    T = dataset.variables['temperature'][:]
    Td = dataset.variables['dewpoint'][:]
    u = dataset.variables['u_wind'][:]
    v = dataset.variables['v_wind'][:]
    
    try:
        lcl_pressure, lcl_temperature = mpcalc.lcl(p[0], T[0], Td[0])
    except:
        lcl_pressure, lcl_temperature = np.nan * p.units, np.nan * T.units
   
    try:
        lfc_pressure, lfc_temperature = mpcalc.lfc(p, T, Td)
    except:
        lfc_pressure, lfc_temperature = np.nan * p.units, np.nan * T.units

        
    return (lcl_pressure.m, lcl_temperature.m, lfc_pressure.m, lfc_temperature.m, T[0].m, Td[0].m)

start_date = datetime(2015, 1, 1, 0)
end_date = datetime(2017, 1, 1, 0)
index = pd.date_range(start_date, end_date, freq='D')
columns = ['LCL Pressure', 'LCL Temperature', 'LFC Pressure', 'LFC Temperature', 'sfc Temperature', 'sfc Dewpoint']

df = pd.DataFrame(index=index, columns=columns)

df = df.apply(get_sounding_parameters, axis=1)

df

N = 30
import matplotlib.dates as mdates
years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
yearsFmt = mdates.DateFormatter('%Y')
# Setup figure and axes
# Generally plots is ~1.33x width to height (10,7.5 or 12,9)
fig = plt.figure(figsize=(10,7.5))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

# Set labels and tick sizes
ax2.set_xlabel(r'Date', fontsize=20)
ax1.set_ylabel(r'LCL Pressure', fontsize=20)
ax2.set_ylabel(r'LCL Temperature', fontsize=20)
# Turns off chart clutter

# Turn off top and right tick marks 
ax1.get_xaxis().tick_bottom()  
ax1.get_yaxis().tick_left()  

ax2.get_xaxis().tick_bottom()  
ax2.get_yaxis().tick_left()      

# Plotting
ax1.scatter(df.index, df['LCL Pressure'], s =2)
ax2.scatter(df.index, df['LCL Temperature'], s =2, color='tab:red')
ax1.plot(df.index, np.convolve(df['LCL Pressure'], np.ones((N,))/N, mode='same'))
ax2.plot(df.index, np.convolve(df['LCL Temperature'], np.ones((N,))/N, mode='same'), color='tab:red')

ax1.set_ylim(1000, 600)
ax2.set_ylim(-30, 30)

ax1.xaxis.set_major_locator(years)
ax1.xaxis.set_major_formatter(yearsFmt)
ax1.xaxis.set_minor_locator(months)
ax2.xaxis.set_major_locator(years)
ax2.xaxis.set_major_formatter(yearsFmt)
ax2.xaxis.set_minor_locator(months)

N = 30
import matplotlib.dates as mdates
years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
yearsFmt = mdates.DateFormatter('%Y')
# Setup figure and axes
# Generally plots is ~1.33x width to height (10,7.5 or 12,9)
fig = plt.figure(figsize=(10,7.5))
ax1 = plt.subplot(111)

# Set labels and tick sizes
ax1.set_xlabel(r'Date', fontsize=20)
ax1.set_ylabel(r'Surface', fontsize=20)

# Turns off chart clutter

# Turn off top and right tick marks 
ax1.get_xaxis().tick_bottom()  
ax1.get_yaxis().tick_left()    

# Plotting
ax1.scatter(df.index, df['sfc Temperature'], s =2, color='tab:red')
ax1.scatter(df.index, df['sfc Dewpoint'], s =2, color='tab:green')
ax1.plot(df.index, np.convolve(df['sfc Temperature'], np.ones((N,))/N, mode='same'), color='tab:red')
ax1.plot(df.index, np.convolve(df['sfc Dewpoint'], np.ones((N,))/N, mode='same'), color='tab:green')

ax1.set_ylim(-20,40)

ax1.xaxis.set_major_locator(years)
ax1.xaxis.set_major_formatter(yearsFmt)
ax1.xaxis.set_minor_locator(months)
ax2.xaxis.set_major_locator(years)
ax2.xaxis.set_major_formatter(yearsFmt)
ax2.xaxis.set_minor_locator(months)



