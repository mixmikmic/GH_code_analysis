get_ipython().magic('matplotlib inline')
from __future__ import division, print_function

import pandas as pd
import numpy as np
import seaborn as sns
from itertools import product

import warnings
sns.set_context("poster")
sns.set_style('ticks')

filename = "C:/WorkSpace/data/exposure/CAIRNS_Residential_Wind_Exposure_201510_M4.csv"

df = pd.read_csv(filename, sep=",",header=0, index_col=0, skipinitialspace=True)
df.info()

def buildingClass(df):
    thresholds = [0.0, 0.8278, 0.973, 1.147]
    classes = ['C1', 'C2', 'C3', 'C4']
    for thres, cls in zip(thresholds, classes):
        idx = np.where(df['M4'] >= thres)[0]
        df['AS4055_CLASS'][idx] = cls
        
    return df

def classifyBuildingAttribute(df, field, classesfrom, classesto):
    ndf = df.copy()
    for cf, ct in zip(classesfrom, classesto):
        ndf[field][ndf[field]==cf] = ct
    return ndf


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

def vulnCurveCombo(df,  default='dw1'):
    walls = ['Brick', 'Cement', 'Timber']
    ages = ['Pre-1950', '1950-1964', '1965-1979', '1980-1994', 'Post-1995']
    n = 1
    for p in product(ages, walls):
        df["WIND_VULNERABILITY_FUNCTION_ID"][(df.YEAR_BUILT==p[0]) & (df.WALL_TYPE==p[1])] = 'dw{0}'.format(n)
        n+=1
    return ndf

classesfrom = np.unique(df['YEAR_BUILT'])
classesto = ['Pre-1950','Pre-1950','Pre-1950', 
             '1950-1964', '1965-1979', '1980-1994', 'Post-1995']
ndf = classifyBuildingAttribute(df, 'YEAR_BUILT', classesfrom, classesto)
walltypesin = np.unique(df['WALL_TYPE'])
walltypesout = ['Brick', 'Cement', 'Timber', 'Timber']
ndf = classifyBuildingAttribute(ndf, 'WALL_TYPE', walltypesin, walltypesout)

from itertools import product
walls = ['Brick', 'Cement', 'Timber']
ages = ['Pre-1950', '1950-1964', '1965-1979', '1980-1994', 'Post-1994']

ndf = vulnCurveCombo(ndf)

outputfile = "C:/WorkSpace/data/exposure/cairns_prepost_classified_iag.csv"
ndf.to_csv(outputfile, header=True)


df = buildingClass(df)

df = vulnCurve(df, 'dw1')

outputfile = "C:/WorkSpace/data/exposure/cairns_prepost_classified.csv"
#df.to_csv(outputfile, header=True)

df = vulnCurve(df, 'dw2')
outputfile = "C:/WorkSpace/data/exposure/cairns_retrofit_classified.csv"
#df.to_csv(outputfile, header=True)

def autolabel(rects, rotation='horizontal'):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        if np.isnan(height):
            height = 0
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
                ha='center', va='bottom', rotation=rotation)

ax = sns.countplot(x='AS4055_CLASS', data=df, palette='RdBu', hue='YEAR_BUILT',
                   order=['C1', 'C2', 'C3', 'C4'])
autolabel(ax.patches, rotation='vertical')

ax.legend(loc=1)
ax.set_xlabel('AS4055 Classification')

ax = sns.countplot(x='WALL_TYPE', data=ndf, palette='RdBu', hue='YEAR_BUILT')
autolabel(ax.patches, rotation='vertical')

ax.legend(loc=1)

grouped = df.groupby(['AS4055_CLASS', 'YEAR_BUILT'])

100 * grouped.count()['LATITUDE']/len(df)

df.groupby(['YEAR_BUILT']).sum()['REPLACEMENT_VALUE']

df.groupby(['YEAR_BUILT']).count()['LATITUDE']

df['WIND_VULNERABILITY_FUNCTION_ID'] = 'dw3'
outputfile = "C:/WorkSpace/data/exposure/cairns_uniform.csv"
#df.to_csv(outputfile, header=True)

ndf.groupby(['YEAR_BUILT', 'WALL_TYPE']).count()['LATITUDE']

pd.pivot_table(ndf, index='WALL_TYPE', columns='YEAR_BUILT', values='REPLACEMENT_VALUE', aggfunc=len)

