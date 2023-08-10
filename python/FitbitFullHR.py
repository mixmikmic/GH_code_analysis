#!/usr/bin/python      
get_ipython().magic('matplotlib inline')

import fitbit
import matplotlib.pyplot as plt
import gather_keys_oauth2 as Oauth2
import numpy as np
import datetime
import pandas as pd
import csv
import re
from IPython.display import display

"""provide your credentials for OAuth2.0"""
USER_ID = 'your USER_ID' # should look something like this: '123A4B'
CLIENT_SECRET = 'your CLIENT_SECRET' # should look something like this: 'c321fvdc59b4cc62156n9luv20k39072'

"""obtain access and refresh tokens"""
server = Oauth2.OAuth2Server(USER_ID, CLIENT_SECRET)
server.browser_authorize()
 
ACCESS_TOKEN = server.oauth.token['access_token']
REFRESH_TOKEN = server.oauth.token['refresh_token']
 
"""complete authorization"""
auth2_client = fitbit.Fitbit(USER_ID, CLIENT_SECRET, oauth2=True, access_token=ACCESS_TOKEN, refresh_token=REFRESH_TOKEN)

start_date = datetime.date(2016, 2, 13)
end_date = datetime.date(2016, 2, 17) # end_date not inclusive

"""function to iterate over dates (resource: http://stackoverflow.com/a/1060330)"""

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)): # for loop through dates in date range
        yield start_date + datetime.timedelta(n) # return date

# adjust your path depending on where you want to store your data
path='/Users/amandasolis/Dropbox/Projects/python-fitbit/Data/'

"""Heart Rate Time Series Function -- Detail Level = 1 second"""

def HR_SeriesSeconds(start_date, end_date):
    HR_df = pd.DataFrame([]) # initialize an empty data frame
    # create index of all 86400 seconds
    full_index=[]
    for i in range(0,86400):
        sec=str(datetime.timedelta(seconds=i))
        full_index.append(sec.zfill(8)) # pad times to have format 00:00:00 rather than 0:00:00
    for single_date in daterange(start_date, end_date): # loop through dates in date range
        date = single_date.strftime("%Y-%m-%d")
        # access Fitbit heart rate log and save as data frame:
        fitbit_stats = auth2_client.intraday_time_series('activities/heart', base_date=date, detail_level='1sec')
        stats = fitbit_stats['activities-heart-intraday']['dataset']
        HR=pd.DataFrame(stats)
        try: # try the following:
            indexed_HR = HR.set_index(HR['time'])
        except KeyError: # if there is no HR['time'] data for this date, create an empty row:
            filler=pd.DataFrame(np.empty(86400, dtype=object))
            full_filler=filler.reindex(full_index) # reindex with the full_index of times from the Calorieseries function 
            filler=full_filler.transpose()
            filler=filler.rename(index={0: str(date)}) # rename the index (row name) with the corresponding date
            HR_df = HR_df.append(filler, ignore_index=False) # add this row to the data frame initialized before the for loop
        else:
            del indexed_HR['time']
            full_HR=indexed_HR.reindex(full_index) # reindex with the full_index of times
            full_HR=full_HR.interpolate() # interpolate missing values
            full_HR=full_HR.fillna(method='bfill') # back fill rows that start with nan
            HR2=full_HR.transpose()
            HR2=HR2.rename(index={'value': str(date)}) # rename the index (row name) with the corresponding date
            HR_df = HR_df.append(HR2, ignore_index=False) # add this row to the data frame initialized before the for loop
    
    # the 'name' variable reflects the folder and file name we are saving the full data frame to:
    name='HR_Second_Timeseries/HR_Second_Timeseries_', str(start_date),'.csv'
    name="".join(name)
    # adding in the full path saves the file to the exact location specified earlier:
    full_path=path, name
    full_path="".join(full_path)
    HR_df.to_csv(full_path)
    
    # save 'HR_df1' as a global variable to explore directly in this notebook
    global HR_Seconds_df1
    HR_Seconds_df1 = HR_df
    return HR_Seconds_df1

HR_SeriesSeconds(start_date,end_date) # obtain heart rate time-series data

time=pd.to_datetime(full_index)
plt.plot(time, HR_Seconds_df1.iloc[1]) # plot one date of heart rate data
plt.gcf().autofmt_xdate()

