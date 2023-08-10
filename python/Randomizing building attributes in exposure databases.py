get_ipython().magic('matplotlib inline')
from __future__ import division, print_function
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

sns.set_context("poster")
sns.set_style('ticks')

filename = "C:/WorkSpace/data/exposure/CAIRNS_Residential_Wind_Exposure_201510_M4.csv"
df = pd.read_csv(filename, sep=",", header=0, index_col=0, skipinitialspace=True)
df.info()

def buildingClass(df):
    thresholds = [0.0, 0.8278, 0.973, 1.147]
    classes = ['C1', 'C2', 'C3', 'C4']
    for thres, cls in zip(thresholds, classes):
        idx = np.where(df['M4'] >= thres)[0]
        df['AS4055_CLASS'][idx] = cls
        
    return df

def vulnCurve(df, default='dw1'):
    classes = ['C1', 'C2', 'C3', 'C4']
    curves = ['dw3', 'dw4', 'dw5', 'dw6']
    # Set all to be default curve to begin with
    df['WIND_VULNERABILITY_FUNCTION_ID'] = default
    filter = df['YEAR_BUILT'].map(lambda x: x not in ['1982 - 1996', '1997 - present'])
    for cls, curve in zip(classes, curves):
        idx = np.where(df['AS4055_CLASS'] == cls)[0]
        df['WIND_VULNERABILITY_FUNCTION_ID'][idx] = curve

    df['WIND_VULNERABILITY_FUNCTION_ID'][filter] = default
    return df

df = buildingClass(df)
df = vulnCurve(df, 'dw1')

def randomize(df, byfield, attribute):
    newdf = df.copy()
    newdf[attribute] = newdf.groupby(byfield)[attribute].transform(np.random.permutation)
    return newdf

newdf = randomize(df, 'SUBURB_2015', 'WIND_VULNERABILITY_FUNCTION_ID')

fig, (ax0, ax1) = plt.subplots(1, 2)

colors = {'dw1':'r', 'dw3':'k', 'dw4':'b', 'dw5':'g', 'dw6':'y'}
minLon = newdf.LONGITUDE.min()
maxLon = newdf.LONGITUDE.max()
minLat = newdf.LATITUDE.min()
maxLat = newdf.LATITUDE.max()


m0 = Basemap(projection='cyl', llcrnrlon=minLon, llcrnrlat=minLat, 
            urcrnrlon=maxLon, urcrnrlat=maxLat, resolution='h', ax=ax0)
m0.drawcoastlines()
m0.drawstates()
m0.drawcountries()
m0.scatter(df.LONGITUDE, df.LATITUDE, 
           c=df['WIND_VULNERABILITY_FUNCTION_ID'].apply(lambda x: colors[x]), 
           alpha=0.25, edgecolors=None, s=4)

m1 = Basemap(projection='cyl', llcrnrlon=minLon, llcrnrlat=minLat, 
            urcrnrlon=maxLon, urcrnrlat=maxLat, resolution='h', ax=ax1)
m1.drawcoastlines()
m1.drawstates()
m1.drawcountries()
m1.scatter(newdf.LONGITUDE, newdf.LATITUDE, 
           c=newdf['WIND_VULNERABILITY_FUNCTION_ID'].apply(lambda x: colors[x]), 
           alpha=0.25, edgecolors=None, s=4)

ax0.set_title("Original exposure")
ax1.set_title("Randomized exposure (by suburb)")

plt.show()

outputfile = "C:/WorkSpace/data/exposure/cairns_retrofit_classified_shuffled.csv"
df.to_csv(outputfile, header=True)

datadict = newdf.to_dict('list')

