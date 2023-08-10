import pandas as pd
wdf = pd.read_csv("..\\data\\raw\\usgs-pemi-flow.txt",sep='\t',comment='#')
wdf.head()

wdf = wdf.drop([0])
wdf.head()

wdf = wdf.rename(index=str, columns={"tz_cd": "timezone","65833_00060": "discharge", 
                                     "65833_00060_cd": "discharge_info",
                                    "65834_00065": "gage_height",
                                    "65834_00065_cd": "gage_height_info",
                                    "65836_00045": "precip",
                                    "65836_00045_cd": "precip_info",})
wdf = wdf.set_index('datetime')
wdf.head()

wdf.describe()

wdf.discharge_info.unique()

wdf.gage_height_info.unique()

wdf.precip_info.unique()

# What are the data types for each column?
wdf.dtypes

# How many nans in each column?
wdf.isnull().sum()

# Add an ice column 
#wdf.loc[wdf['discharge'] == 'Ice']
import numpy as np
wdf['ice'] = np.where(wdf['discharge']=='Ice', 1,0)
wdf.head()

# Number of ice rows 
wdf.ice.sum()

# Expected null columns
num_ice_cols = 2802 + 849
print(str(num_ice_cols))

# Convert data in these columns from object to numeric data type
int_df = wdf
int_df.discharge = pd.to_numeric(wdf.discharge,errors='coerce')
int_df.gage_height = pd.to_numeric(wdf.gage_height,errors='coerce')
int_df.precip = pd.to_numeric(wdf.precip,errors='coerce')
int_df.head()

# How many nans in each column?
int_df.isnull().sum()

# Interpolate the data to remove NaNs
int_df = int_df.interpolate()

# How many nans in each column?
int_df.isnull().sum()

# Normalize data to max value 
int_df['gage_height'] = int_df.gage_height/int_df['gage_height'].max()
int_df['discharge'] = int_df.discharge/int_df['discharge'].max()
int_df.head()

import matplotlib.pyplot as plt
plt.show()

# Plot data 
import matplotlib
int_df.discharge.plot()
int_df.ice.plot()
int_df.gage_height.plot()

int_df.describe(include='all')

