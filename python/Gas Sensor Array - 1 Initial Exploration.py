get_ipython().magic('matplotlib inline')

# numbers
import numpy as np
import pandas as pd

# stats
import statsmodels.api as sm
import scipy.stats as stats

# plots
import matplotlib.pyplot as plt
import seaborn as sns

# utils
import os, re, io
from pprint import pprint

b1_files = [j for j in os.listdir('data/twin_gas_sensors') if 'B1' in j]
print len(b1_files)

all_files = [j for j in os.listdir('data/twin_gas_sensors')]
print len(all_files)

print b1_files[0]

def sensor_df(fname):
    prefix = 'data/twin_gas_sensors/'

    sensors = ['Sensor '+str(j) for j in range(8)]
    sensor_columns = ['Time'] + sensors

    df = pd.read_csv(prefix+fname,delimiter='\t',header=None,names=sensor_columns)
    return df

sdf1 = sensor_df(b1_files[0])

print sdf1.head()

for j in range(8):
    seriesname = 'Sensor '+str(j)
    plt.plot(sdf1['Time'],sdf1[seriesname], label=seriesname)

plt.title("Gas Sensor Array: Experiment 1")
plt.xlabel('Elapsed Experimental Time (s)')
plt.ylabel('Sensor Resistance (kOhm)')
plt.legend(bbox_to_anchor=(1.3, 0.8))
plt.show()



