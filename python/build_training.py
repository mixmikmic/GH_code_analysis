get_ipython().run_line_magic('matplotlib', 'inline')
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time
import geopandas
from IPython.display import display # display(df) shows dataframe in html formatting
from pandas.tseries.holiday import USFederalHolidayCalendar

print(sys.version)
print(np.__version__)
print(pd.__version__)

## Example for displaying more of a dataframe
# with pd.option_context('display.max_rows',300,'display.max_columns', 100):
#     display(df)

# default paths for raw and clean data files
data_path = './Fire_Department_Calls_for_Service.csv'
clean_save_path = './clean_sf_fire.csv'

data_types={'Incident Number':np.int64, 'Call Type':str, 'On Scene DtTm':str,
            'Received DtTm': str, 'Zipcode of Incident':np.float64, 'Unit Type':str,
            'Unit sequence in call dispatch':np.float64,'Location':str, 'RowID':str}
date_cols = ['Received DtTm']

df = pd.read_csv(data_path, dtype=data_types, usecols=data_types.keys(), parse_dates=date_cols, na_values='None')

with pd.option_context('display.max_rows',300,'display.max_columns', 100):
    display(df.head(200))

# total dispatches and unique incidents
print('Dispatches        :', len(df))
print('Unique Incidents  :', df['Incident Number'].nunique())

# Removing records without an on-scene time
df = df[~pd.isnull(df['On Scene DtTm'])]

# total dispatches and unique incidents
print('Dispatches        :', len(df))
print('Unique Incidents  :', df['Incident Number'].nunique())

# Filtering down to the top three incident call types and medic/private units
call_type_keep = ['Medical Incident', 'Traffic Collision', 'Structure Fire']
unit_type_keep = ['MEDIC','PRIVATE']
df = df[(df['Call Type'].isin(call_type_keep)) & (df['Unit Type'].isin(unit_type_keep))]

# total dispatches and unique incidents
print('Dispatches        :', len(df))
print('Unique Incidents  :', df['Incident Number'].nunique())

# Splitting lat/lon field in to two numeric fields
df['lat'], df['lon'] = zip(*df.Location.map(lambda x: [float(val) for val in x.strip('()').split(',')]))

# Getting list of US Federal Holidays (includes observed)
holidays = USFederalHolidayCalendar().holidays(start='2000-01-01', end='2018-01-01')

# Splitting received datetime into separate parts, including flag for weekends
df['year'], df['month'], df['day_of_month'], df['hour_of_day'], df['day_of_year'], df['week_of_year'], df['day_of_week'], df['is_weekend'],  =     zip(*df['Received DtTm'].map(lambda val: [val.year, val.month, val.day, val.hour, val.dayofyear, val.week, val.weekday(), val.weekday() in [5,6]]))
    
df['is_holiday'] = df['Received DtTm'].isin(holidays)

with pd.option_context('display.max_rows',300,'display.max_columns', 100):
    display(df.head(200))

# Loading ZCTA shape file with geopandas
gdf = geopandas.read_file('./sf_zcta.shp')

def get_zcta(point):
    '''
    Takes a geopandas/shapely longitude/latitude point object and returns
    the US Census Zip Code Tabulation Area containing it.
    INPUT: Point must have longitude first, since it expects an x,y coordinate
    OUTPUT: The ZCTA code, or None if not found in San Francisco County
    '''
    zcta = gdf.ZCTA5CE10[gdf.geometry.contains(point)]
    if len(zcta) > 0:
        return zcta.values[0]
    else:
        return None

# Finding ZCTA for each point
df['zcta'] = df.apply(lambda row: get_zcta(geopandas.geoseries.Point(row.lon, row.lat)), axis=1)

df.head()

# We need to remove records that are not in a ZCTA. These are likely border cases where a unit was dispatched
# outside of SF for some reason. Most of these cases also have no Zip in the data
print(len(df[pd.isnull(df['zcta'])]))
df = df[~pd.isnull(df['zcta'])]

# Setting region to be the ZCTA. This is to keep the region field generic in case we choose to use some other kind
# of region mapping in the future.
df['region'] = df.zcta

temp = df[~pd.isnull(df['Zipcode of Incident'])].copy()
temp['Zipcode of Incident'] = temp['Zipcode of Incident'].map(lambda x: str(int(x)))
temp['is_match'] = temp['Zipcode of Incident'] == temp['zcta']
len(temp.is_match)

sum(temp.is_match)

temp2 = temp[~temp.is_match]
len(temp2)

temp2.head(200)

# total dispatches and unique incidents
print('Dispatches        :', len(df))
print('Unique Incidents  :', df['Incident Number'].nunique())

with pd.option_context('display.max_rows',300,'display.max_columns', 100):
    display(df.head(200))

# filtering for fields that will be retained in training set
keep_fields = ['year', 'month', 'day_of_month', 'hour_of_day', 'day_of_year',
               'week_of_year', 'day_of_week', 'is_weekend', 'is_holiday',
               'region']

df = df[keep_fields]

# creating a unique region and time field for grouping on an hourly basis in each region
df['region_time'] = df.apply(lambda row: '-'.join([str(row.region), str(row.year), str(row.month), str(row.day_of_month), str(row.hour_of_day)]), axis=1)

# Creating dictionary of the number of dispatches in each region/hour tuple
# The 'year' field is arbitrary, I just needed it to return a series of counts instead of a dataframe
dispatch_counts = dict(df.groupby(['region_time'])['year'].count())

# Verifying that count matches
print(sum(dispatch_counts.values()))
print(len(df))

# Dropping duplicate region_time rows
df.drop_duplicates(subset='region_time', inplace=True)

# assigning dispatch counts and checking total count again
df['dispatch_count'] = df.apply(lambda row: dispatch_counts[row.region_time], axis=1)

sum(df.dispatch_count)

len(df)

with pd.option_context('display.max_rows',300,'display.max_columns', 100):
    display(df.head(200))

# df.to_csv('./sf_ems_clean.csv')

