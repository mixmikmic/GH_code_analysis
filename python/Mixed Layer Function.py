from datetime import datetime

import matplotlib.pyplot as plt

import metpy.calc as mpcalc
from metpy.io import get_upper_air_data
from metpy.plots import SkewT
from metpy.units import units
import numpy as np
from metpy.constants import kappa
get_ipython().magic('matplotlib inline')

# Download and parse the data
dataset = get_upper_air_data(datetime(1999, 5, 4, 0), 'OUN')

p = dataset.variables['pressure'][:]
T = dataset.variables['temperature'][:]
Td = dataset.variables['dewpoint'][:]
u = dataset.variables['u_wind'][:]
v = dataset.variables['v_wind'][:]

def mixed_layer(p, datavar, bottom=None, depth=100*units.hPa, interpolate=True):
    p_layer, datavar_layer = mpcalc.get_layer(p, datavar, bottom, depth, interpolate)
    actual_depth = abs(p_layer[0] - p_layer[-1])
    datavar_mean = (1./actual_depth.m) * np.trapz(datavar_layer, p_layer) * datavar.units
    return datavar_mean

def mixed_parcel(p, T, Td, parcel_start_pressure=p[0], bottom=None, depth=100*units.hPa, interpolate=True):
    
    # Get the variables in the layer
    p_layer, T_layer = mpcalc.get_layer(p, T, bottom, depth, interpolate)
    p_layer, Td_layer = mpcalc.get_layer(p, Td, bottom, depth, interpolate)
    
    # Calculate the potential temperature and mixing ratio over the layer
    theta = mpcalc.potential_temperature(p, T)
    mixing_ratio = mpcalc.saturation_mixing_ratio(p, Td)
    
    # Mix the variables over the layer
    mean_theta = mixed_layer(p, theta, bottom, depth, interpolate)
    mean_mixing_ratio = mixed_layer(p, mixing_ratio, bottom, depth, interpolate)
    
    # Convert back to temperature
    mean_temperature = mean_theta / mpcalc.potential_temperature(parcel_start_pressure, 1 * units.degK).m
    
    # Convert back to dewpoint
    mean_vapor_pressure = mpcalc.vapor_pressure(parcel_start_pressure, mean_mixing_ratio)
    mean_dewpoint = mpcalc.dewpoint(mean_vapor_pressure)
    
    return parcel_start_pressure, mean_temperature.to(T.units), mean_dewpoint.to(Td.units)

plt.plot(p, T)
plt.xlim(1000, 800)
plt.ylim(10, 25)

mixed_layer(p, T)

pl, tl = mpcalc.get_layer(p, T, bottom=3000*units.m, depth=2000*units.m, interpolate=True)

plt.plot(pl, tl, marker='o')

mixed_parcel(p, T, Td, depth=500*units.hPa)





fig = plt.figure(figsize=(9, 9))
skew = SkewT(fig, rotation=45)

# Plot the data using normal plotting functions, in this case using
# log scaling in Y, as dictated by the typical meteorological plot
skew.plot(p, T, 'r')
skew.plot(p, Td, 'g')
skew.plot_barbs(p, u, v)
skew.ax.set_ylim(1000, 100)
skew.ax.set_xlim(-40, 60)

# Calculate LCL height and plot as black dot
lcl_pressure, lcl_temperature = mpcalc.lcl(p[0], T[0], Td[0])
skew.plot(lcl_pressure, lcl_temperature, 'ko', markerfacecolor='black')

# Calculate full parcel profile and add to plot as black line
prof = mpcalc.parcel_profile(p, T[0], Td[0]).to('degC')
skew.plot(p, prof, 'k', linewidth=2)

# Example of coloring area between profiles
greater = T >= prof
skew.ax.fill_betweenx(p, T, prof, where=greater, facecolor='blue', alpha=0.4)
skew.ax.fill_betweenx(p, T, prof, where=~greater, facecolor='red', alpha=0.4)

# An example of a slanted line at constant T -- in this case the 0
# isotherm
skew.ax.axvline(0, color='c', linestyle='--', linewidth=2)

# Add the relevant special lines
skew.plot_dry_adiabats()
skew.plot_moist_adiabats()
skew.plot_mixing_lines()

# Show the plot
plt.show()

