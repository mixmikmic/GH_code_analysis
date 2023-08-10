import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
get_ipython().magic('matplotlib inline')
import seaborn as sns; sns.set()

# Load the data
time_count = pd.read_csv('jsons_summary.csv')
time_count.columns = ["", "response_time_stamp", "veh_count", "veh_str_len", "filename"]
time_count = time_count[['response_time_stamp', 'veh_count', 'veh_str_len', 'filename']]
time_count.head()

# Change the response_time_stamp type from sting to datetime
time_count['response_time_stamp'] = pd.to_datetime(time_count.response_time_stamp)
time_count['source'] = time_count.filename.str[11:]
time_count.sort_values(['source','response_time_stamp'],inplace=True)
time_count.head()

time_count['time'] = time_count.response_time_stamp.apply(str)
time_count.head()

time1 = time_count['time']
time = map(lambda x:x[11:13], time1)
date = map(lambda x:x[:10], time1)
time_count['hour'] = time
time_count['date'] = date
time_count.head()

time_count['weekday'] = time_count['response_time_stamp'].apply(datetime.weekday)
time_count.head()

count_hour = time_count.groupby(['weekday','hour']).sum()
count_hour.head()

count_hour.to_csv('count_hour.csv')

count_hour = pd.read_csv('count_hour.csv')
count_hour = count_hour[['weekday', 'hour', 'veh_count']]
count_hour.head()

# Plot the data
weekdaylist = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']
fig = plt.figure(figsize=(16,8))

for i in range(0,7):
    count_tmp = count_hour[count_hour['weekday'] == i]
    #fig = plt.figure(figsize=(12,8))
    ctdatetmp = count_tmp.hour
    ctbustmp  = count_tmp.veh_count
    ax = fig.add_subplot(2,4,i+1)
    ax.bar(ctdatetmp, ctbustmp, width=1)
    ax.set_xlabel('Hour')
    ax.set_ylabel('Record Count')
    ax.set_xlim(0,24)
    ax.set_title('Bus Record Count For Every Hour In '+ weekdaylist[i]+'\n')

fig.tight_layout()

