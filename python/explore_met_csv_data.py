from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

get_ipython().magic('matplotlib inline')

df = pd.read_csv('data_gov_sg_met_v1/wind-direction_2016_12_c20170530.csv.gz')
df.info()

df['timestamp_sgt'] = df['timestamp_sgt'].astype('datetime64[ns]')  # convert to datetime64 type
df.info()

df.head()

df.tail()

# Read data for each variable and store in dictionary
df_dict = {}  # dictionary in which to store DataFrames
for variable in ['rainfall', 'wind-speed', 'wind-direction']:
    df = pd.DataFrame()  # initialise df as an empty DataFrame
    filenames = glob('data_gov_sg_met_v1/{}_*.csv.gz'.format(variable))
    for filename in filenames:
        temp_df = pd.read_csv(filename)  # read data from file
        temp_df = temp_df.rename(columns={'value': variable})  # rename 'value' column
        temp_df['timestamp_sgt'] = temp_df['timestamp_sgt'].astype('datetime64[ns]')  # convert to datetime64
        df = df.append(temp_df, ignore_index=True)  # append to df
        print('Read {}'.format(filename))
    df_dict[variable] = df  # store df in dictionary

# Union across the different variables
outer_df = df_dict['rainfall']  # initialise with rainfall data
for variable in ['wind-speed', 'wind-direction']:
    outer_df = outer_df.merge(df_dict[variable], how='outer', on=['station_id', 'timestamp_sgt'])
outer_df.info()

outer_df.head()

outer_df.tail()

# Intersection across the different variables
inner_df = df_dict['rainfall']  # initialise with rainfall data
for variable in ['wind-speed', 'wind-direction']:
    inner_df = inner_df.merge(df_dict[variable], how='inner', on=['station_id', 'timestamp_sgt'])
inner_df.info()

inner_df.head()

inner_df.tail()

outer_df.groupby('station_id').describe(percentiles=[0.5,]).sort_index()

# Get index of stations with no wind-speed data
temp_df = outer_df.groupby('station_id')['wind-speed'].count()  # number of data points per station
temp_df = temp_df[temp_df == 0]  # select stations with no data
no_wind_stations = list(temp_df.index)
# Drop stations with no wind-speed data
temp_df = outer_df.set_index('station_id')
has_wind_df = temp_df.drop(no_wind_stations).sort_index()

# Wind-speed
fig, ax = plt.subplots(figsize=(16, 6))
sns.violinplot(x=has_wind_df.index, y=has_wind_df['wind-speed'], cut=0, ax=ax)
ax.set_ylabel('Wind-speed, knots')

# Rainfall
fig, ax = plt.subplots(figsize=(16, 6))
sns.violinplot(x=has_wind_df.index, y=has_wind_df['rainfall'], cut=0, ax=ax)
ax.set_ylabel('5-min rainfall, mm')

# Rainfall - excluding zero
fig, ax = plt.subplots(figsize=(16, 6))
temp_df = has_wind_df[has_wind_df['rainfall'] > 0]
sns.violinplot(x=temp_df.index, y=temp_df['rainfall'], cut=0, ax=ax)
ax.set_ylabel('5-min rainfall (excluding zero), mm')

# Select data for station S06, and index by timestamp_sgt
s06_df = outer_df[outer_df['station_id'] == 'S06'].set_index('timestamp_sgt')

# Prepare axes
fig, ax = plt.subplots(figsize=(16, 6))
ax1 = ax.twinx()
# Left axis - windspeed timeseries
s06_df.plot(y='wind-speed', linewidth=0.2, c='r', ax=ax)
ax.legend(loc=2)
ax.set_ylabel('Wind-speed, knots')
ax.set_ylim([0, 20])
ax.grid(False)
# Right axis - rainfall timeseries scatter
ax1.scatter(x=s06_df.index, y=s06_df['rainfall'], marker='x', s=10)
ax1.legend(loc=1)
ax1.set_ylabel('5-min rainfall, mm')
ax1.set_ylim([0, 10])
ax1.grid(False)

# Quick look at relationship between wind-speed and rainfall for one station
sns.jointplot(s06_df['wind-speed'], s06_df['rainfall'], kind='scatter', dropna=True,
              marker='x', s=10)

# What happens if times with zero rain are excluded?
s06_has_rainfall = s06_df[s06_df['rainfall'] > 0]
sns.jointplot(s06_has_rainfall['wind-speed'], s06_has_rainfall['rainfall'], kind='scatter', dropna=True,
              marker='x', s=10)

