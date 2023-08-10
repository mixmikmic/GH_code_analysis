import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.size'] = 16
plt.rcParams['figure.figsize'] = 10, 5

pressure = np.loadtxt('pressure.30733')

plt.plot(pressure[:,0], pressure[:,1])
plt.show()

#Check the time scale
print pressure[21,0], pressure[22,0]

def smooth(data, N=20):
    return np.convolve(data, np.ones(N)/float(N))[(N-1):]

plt.plot(pressure[:,0], pressure[:,1])
plt.plot(pressure[:,0], smooth(pressure[:,1], 50), color='r')
plt.show()

smthpres = smooth(pressure[:,1], 50)

fuelling = np.loadtxt('fuel.30733')

plt.plot(fuelling[:,0], fuelling[:,1])
plt.show()

pheat = np.loadtxt('heating_power.30733')
plt.plot(pheat[:,0], pheat[:,1])
plt.show()

import pandas as pd
from scipy.interpolate import interp1d

common_time = np.arange(1.5, 7.5, 0.01)

f_interp = interp1d(fuelling[:,0], fuelling[:,1])
new_fuel = f_interp(common_time)
plt.plot(common_time, new_fuel)
plt.show()

h_interp = interp1d(pheat[:,0], pheat[:,1])
new_power = h_interp(common_time)
plt.plot(common_time, new_power)
plt.show()

press_interp = interp1d(pressure[:,0], smthpres)
new_press = press_interp(common_time)
plt.plot(common_time, new_press)
plt.show()

#Declare the name with unitsstrtime = 'Time [s]'
strTime = 'Time [s]'
strFuel = 'Fuelling [$10^{22}$e/s]'
strPow = 'Power [MW]'
strPres = 'Pressure [$10^{23}$e$m^{-2}$]'

#Pass a dictionary as input
d = {strTime: common_time,
     strFuel: new_fuel*1e-22,
     strPow: new_power*1e-6,
     strPres: new_press*1e-23}

df = pd.DataFrame(data=d)

df.head()

col_order = [strTime, strFuel, strPow, strPres]

df = df[col_order]

df.head()

df.plot(kind='line', x=strTime, figsize=(10, 5), fontsize=14)
plt.show()



df.plot(kind='line', x=strTime, figsize=(10, 5), fontsize=14)
plt.show()

ldf = df.copy(deep=True)

ldf.dropna(axis=0, inplace=True)

ldf.plot(kind='line', x=strTime, figsize=(10,5))
plt.show()

ldf.head()

ldf[strPow] = np.log(ldf[strPow])
ldf[strFuel] = np.log(ldf[strFuel])
ldf[strPres] = np.log(ldf[strPres])

ldf.head()

#Define a fitting function
def fit_func( (lfuel, lheat), k, a, b):
    return k + a*lfuel + b*lheat

from scipy.optimize import *

popt, pcov = curve_fit(fit_func, (ldf[strFuel], ldf[strPow]), ldf[strPres])

#The optimised parameters of the fit
print "popt:", popt
#One standard deviation errors on the parameters.
perr = np.sqrt(np.diag(pcov))
print "perr:", perr
#The covariance matrix of the parameters
print "pcov:", pcov

plt.scatter(ldf[strPres], fit_func( (ldf[strFuel], ldf[strPow]), *popt))
plt.show()

plt.scatter(ldf[strTime], ldf[strPres], label='Pressure', color='blue')
plt.scatter(ldf[strTime], fit_func( (ldf[strFuel], ldf[strPow]), *popt), label='Fit', color='red')
plt.legend()
plt.show()



