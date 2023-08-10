get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import geopandas as gpd
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
pylab.rcParams['figure.figsize'] =10, 8

vtroad= gpd.read_file('/VT_Road_Centerline.shp', vfs='zip://VT_Road_Centerline.zip')

vtroad.head()

vtroad.plot()

vtroad.info()

vtroad.describe()

sns.countplot(vtroad['PD'], palette="husl")
plt.title('Count of Road Direction')
plt.xlabel('Direction')
plt.ylabel('Count')
plt.show()

sns.countplot(vtroad['SURFACETYP'], palette="Set1")
plt.title('Count of Surface Types')
plt.xlabel('Surface Type')
plt.ylabel('Count')
plt.show()

types = vtroad[['SURFACETYP', 'PD']].groupby(['SURFACETYP'], as_index=False).count()
types.columns = ['Type','Count']
types = types.sort_values(by='Count',ascending=0)
types['Percentage'] = types['Count'] / types['Count'].sum()
types.head(6)

data = vtroad[(vtroad['SURFACETYP'] == 2) | (vtroad['SURFACETYP'] == 3)]
data.plot()
plt.title("All the Dirt and Roads")
plt.show()

vtroad['SPEEDLIMIT'] = np.array(vtroad['SPEEDLIMIT'], dtype=np.float)
pd.unique(vtroad['SPEEDLIMIT'])

sns.countplot(vtroad['SPEEDLIMIT'])
plt.title('Count of Speed Limits')
plt.show()

# return the number of null values
vtroad.SPEEDLIMIT.isnull().sum()

sns.countplot(vtroad['AOTCLASS'])

AOTclass = vtroad[['AOTCLASS', 'PD']].groupby(['AOTCLASS'], as_index=False).count()
AOTclass.columns = ['Type','Count']

# remove all the outliers
AOTclass = AOTclass[(AOTclass['Count'] > 5)]

# Sort and add percentage
AOTclass = AOTclass.sort_values(by='Count', ascending=0)
AOTclass['Percentage'] = AOTclass['Count'] / AOTclass['Count'].sum()

#change from numeric classifications to a text one
AOTclass.loc[(AOTclass['Type'] == 1), 'Type'] = "Town Highway Class 1"
AOTclass.loc[(AOTclass['Type'] == 2), 'Type'] = "Town Highway Class 2"
AOTclass.loc[(AOTclass['Type'] == 3), 'Type'] = "Town Highway Class 3"
AOTclass.loc[(AOTclass['Type'] == 4), 'Type'] = "Town Highway Class 4"
AOTclass.loc[(AOTclass['Type'] == 8), 'Type'] = "Private road"
AOTclass.loc[(AOTclass['Type'] == 30), 'Type'] = "State Highway"
AOTclass.loc[(AOTclass['Type'] == 40), 'Type'] = "US Highway"
AOTclass

sns.barplot(x='Type', y='Count', data=AOTclass, palette="Set2")
plt.title('Count of Roads by Type')
plt.xlabel("Type")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

vtroad.shape

# takes the first 1,000 rows of data
truncroad = vtroad[0:1000]
truncroad.shape

truncroad.describe()

sns.boxplot(x='SPEEDLIMIT', y='AOTMILES', data=truncroad)

speed =  vtroad[(vtroad['SPEEDLIMIT'].notnull())]
speed.shape

pd.unique(speed['SCENICBYWA'])

from random import *
 
randint(1, 25) 

