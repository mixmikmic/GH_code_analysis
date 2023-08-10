import requests
r = requests.get('https://docs.google.com/spreadsheets/d/1lqIWdnmjiZX2LwHZ_5TdPXDOEn8hp-ZkdONlbjA-P1k/export?format=csv&id')
data = r.content

from StringIO import StringIO
import pandas as pd
adb = pd.io.parsers.read_csv(StringIO(data),index_col=0)
adb

import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')

species = 'Benzene'

T = np.linspace(adb.ix[species]['Tmin'],adb.ix[species]['Tmax'])

def Psat(s,T):
    return 10.0**(adb.ix[s]['A'] - adb.ix[s]['B']/(T + adb.ix[s]['C']) )

plt.plot(T,Psat(species,T))
plt.xlabel('Temperature [deg C]')
plt.ylabel('Pressure [mmHg]')
plt.title('Saturation Pressure of ' + species)
plt.grid()

