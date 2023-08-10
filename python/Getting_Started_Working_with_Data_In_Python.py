#Let's get data
get_ipython().system('curl -O http://isalix.hestia.feralhosting.com/2016.csv.gz    ')
get_ipython().system('gunzip 2016.csv.gz')
get_ipython().system('mv 2016.csv ~/data')
get_ipython().system('curl -O http://www1.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt')
get_ipython().system('mv ghcnd-stations.txt ~/data')

# Import Statements
import pandas as pd
import numpy as np
get_ipython().magic('matplotlib inline')

# File Locations
# Change these on your machine!
datadir = "/Users/matthewgee/data/"
weather_data_raw = datadir + "2016.csv"
station_data_raw = datadir + "ghcnd-stations.txt"

get_ipython().magic('pinfo pd.read_table')

weather = pd.read_table(weather_data_raw, sep=",", header=None)
#weather = pd.read_csv(weather_data_raw)
stations = pd.read_table(station_data_raw, header=None)

weather.head()

weather.tail()

weather.shape

weather.dtypes

weather.columns

#Get rid of 4,5,6, & 7
weather.drop([4,5,6,7], axis=1, inplace=True)

weather.head()

weather_cols = ['station_id','date','measurement','value']
weather.columns = weather_cols
weather.columns

weather.describe()

weather.describe(include=['O'])

weather['measurement'].head()

#using . notation
weather.measurement.value_counts(normalize=True)

#subset by row index
weather.measurement[3:10]

#Use the iloc method 
weather.iloc[10:20,2:4]

#Create a boolean series based on a condition
weather['measurement']=='PRCP'

#now pass that series to the datafram to subset it
rain = weather[weather['measurement']=='PRCP']
rain.head()

rain.sort_values('value', inplace=True, ascending=False)

rain.head()

#Let's create a chicago tempurature dataset
chicago = weather[weather['station_id']=='USW00094846']
chicago_temp = weather[(weather['measurement']=='TAVG') & (weather['station_id']=='USW00094846')]
chicago_temp.head()

chicago_temp.sort_values('value').head()

chicago_temp = chicago_temp[chicago_temp.value>-40]
chicago_temp.head()

chicago_temp.value.mean()

chicago_temp.value.describe()

#Apply user defined functions
def ftoc(temp_f):
    return (temp_f-32)*5/9

chicago_temp['TAVG_CEL']=chicago_temp.value.apply(ftoc)
chicago_temp.describe()

chicago_temp['datetime'] = pd.to_datetime(chicago_temp.date, format='%Y%m%d')
chicago_temp.dtypes

chicago_temp.head()

chicago_temps = chicago[chicago.measurement.isin(['TMAX','TMIN','TAVG'])]
chicago_temps.measurement.value_counts()

chicago_temps.groupby('measurement').value.mean()

chicago_temps.groupby('measurement').value.agg(['count','min','max','mean'])

chicago_temps.groupby('measurement').value.mean().plot(kind='bar')

chicago_temps.value.plot()

chicago_temps['datetime'] = pd.to_datetime(chicago_temps.date, format='%Y%m%d')
chicago_temps.index = chicago_temps.datetime
chicago_temps.dtypes
chicago_temps.head()

chicago_temps.groupby('measurement').value.plot(figsize=(15,5))



