import pandas as pd
import numpy as np
get_ipython().magic('matplotlib inline')

df=pd.read_csv("datasets/yellow_tripdata_2013-01.csv")

df = df.loc[(df['pickup_longitude'] > -74.06) & (df['pickup_longitude'] < -73.77) & (df['pickup_latitude'] > 40.61) &  (df['pickup_latitude'] < 40.91)]
df = df.loc[(df['tip_amount'] > 0.0)] #comment this line out to include those who don't pay tips
df = df.reset_index()
df = df.drop('index', 1)
df = df.drop('vendor_id',1)
df = df.drop('rate_code',1)
df = df.drop('store_and_fwd_flag',1)
df = df.drop('fare_amount',1)
df = df.drop('surcharge',1)
df = df.drop('mta_tax',1)
df = df.drop('tolls_amount',1)
df = df.drop('dropoff_latitude',1)
df = df.drop('dropoff_longitude',1)
df = df.drop('passenger_count',1)
df = df.drop('payment_type',1)

def add_data(df):
    df_timestamp = pd.to_datetime(pd.Series(df['pickup_datetime']))
    df['trip_distance']*0.621371 # convert to miles
    df['weekday'] = df_timestamp.dt.weekday_name
    #df['month'] = df_timestamp.dt.month
    df['hour'] = df_timestamp.dt.hour
    #df['day'] = df_timestamp.dt.day
    #df['minutes'] = (df_timestamp.dt.hour)*60 + df_timestamp.dt.minute
    time_spent = pd.to_datetime(df['dropoff_datetime']) - pd.to_datetime(df['pickup_datetime'])
    df['time_spent'] = pd.to_datetime(time_spent).dt.minute
    df['pickup'] = df['pickup_latitude'].map(str) +','+df['pickup_longitude'].map(str)
    return df

df = add_data(df)

df = df.drop('pickup_datetime',1)
df = df.drop('dropoff_datetime',1)
df = df.drop('pickup_longitude',1)
df = df.drop('pickup_latitude',1)

df.head()

# Look into the dataframe 
# First by weekday, then hour then block
# and we will know the average tip amount
# for each weekday > hour > block
# eg. On (day of week) at (hour) on (lat,long) avg tip is $number
def get_avg_tips(df):
    avg_tips = df.groupby(['weekday','hour','pickup']).mean()
    avg_tips = avg_tips.reset_index()
    return avg_tips

df = get_avg_tips(df)

df.shape

df.head()

df.tail()

#df.to_csv('clean-january-2013.csv',sep=',',encoding='utf-8') # both types
df.to_csv('cleaner-january-2013.csv',sep=',',encoding='utf-8') # for only those who gave tips



