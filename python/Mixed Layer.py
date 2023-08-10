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
dataset = get_upper_air_data(datetime(1999, 5, 4, 0), 'OUN', retry_delay=1000)

p = dataset.variables['pressure'][:]
T = dataset.variables['temperature'][:]
Td = dataset.variables['dewpoint'][:]
u = dataset.variables['u_wind'][:]
v = dataset.variables['v_wind'][:]

def mixed_layer(p, T, Td, depth=100*units.hPa, starting_pressure=p[0]):
    
    bottom_pressure = p[0]
    top_pressure = p[0] - depth
    
    inds = (p <= bottom_pressure) & (p >= top_pressure)
    p_interp = p[inds]
    print(p_interp)
    p_interp = np.sort(np.append(p_interp, top_pressure)) * units.hPa
    sort_args = np.argsort(p)
    T = np.interp(p_interp, p[sort_args], T[sort_args]) * units.degC
    Td = np.interp(p_interp, p[sort_args], Td[sort_args]) * units.degC
    p = p_interp
    
    theta = mpcalc.potential_temperature(p, T)
  
    mixing_ratio = mpcalc.saturation_mixing_ratio(p, Td)
    plt.plot(p, theta)
   
    actual_depth = p[-1] - p[0]
    theta_mean = (1./actual_depth.m) * np.trapz(theta, p) * units.kelvin
    mixing_ratio_mean = (1./actual_depth.m) * np.trapz(mixing_ratio, p)
    vapor_pressure_mean = mpcalc.vapor_pressure(starting_pressure, mixing_ratio_mean)
    
    dewpoint_mean = mpcalc.dewpoint(vapor_pressure_mean)
    temperature_mean = theta_mean / mpcalc.potential_temperature(starting_pressure, 1*units.degK).m
    return starting_pressure, temperature_mean.to('degC'), dewpoint_mean

print(mixed_layer(p, T, Td))

mpcalc.virtual_temperature(1 * units.degC, 0.01229)

mpcalc.potential_temperature(959*units.hPa, 20 * units.degC)



mpcalc.potential_temperature(959*units.hPa, 275*units.degK)

278.30809 /( (1000 / 959)**kappa)

278.30809/ mpcalc.potential_temperature(959*units.hPa, 1*units.degK)

(300*units.degK).to('degC')

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

x = [860,960]
y = [302, 299]
print(np.polyfit(x, y, 1))

def integ(x):
    return -0.015*x*x+327.8*x

integ(960)-integ(860)

np.trapz([302,299], [860,960])

pint = [959,931.3,925.,899.3,892.,867.9]
tint = [298.90289633,299.37465065,299.54662685,300.52407484,300.81124879,302.54903507]
np.trapz(tint[::-1], pint[::-1])

plt.plot(pint, tint, marker='o')

27349.019918277005/(959-867.9)

a= np.array([1,2,3])

a

np.append(a, [4])

p.dimensionality

p.dimensionality == {'[length]': -1.0, '[mass]': 1.0, '[time]': -2.0}

leng = 1 * units.m

leng

leng.dimensionality 

leng.units

from time import sleep

for i in range(10):
    print(i)
    dataset = get_upper_air_data(datetime(1999, 5, 4, 0), 'OUN', retry_delay=1000)
    #sleep(0.01)



