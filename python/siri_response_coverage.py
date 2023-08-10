import os
import pandas as pd
from datetime import datetime
from datetime import timedelta
get_ipython().magic('matplotlib inline')
os.chdir('/gpfs2/projects/project-bus_capstone_2016/workspace/share')

data = pd.read_csv('jsons_summary.csv',index_col=0)
data.head(20)

data['source'] = data.filename.str[11:]
data['hour'] = data.response_time_stamp.str[11:13]
data['response_date'] = data['response_time_stamp'].str[:10]
data.sort(['source','response_time_stamp'],inplace=True)
data.reset_index(drop=True,inplace=True)
data['response_time_stamp'] = pd.to_datetime(data.response_time_stamp)
# also calculate the difference from one to the next
data['elapsed'] = data['response_time_stamp'].diff()/timedelta(seconds=1)
data['weekday'] = data['response_time_stamp'].apply(datetime.weekday)

data['elapsed'].hist(range=(0,300),bins=60)

sum(data['elapsed']>300)

sum(data['elapsed']>1800)

data.query('elapsed > 1800').groupby('response_date').size().plot()

data.query('elapsed > 1800').groupby('weekday').size()

data.query('elapsed > 1800 & weekday <= 5').groupby('hour').size()

data[data['response_date']=='2015-11-05'].groupby('hour').size()

gaps_by_date = data.query('elapsed > 1800').groupby('response_date').size()

gaps_by_date.sort()

gaps_by_date[:5]

gaps_by_date[-15:]

for d in data.response_date.unique():
    if d in list(gaps_by_date.index):
        pass
    else:
        print d

data.query('response_date >= "2015-12-01" & response_date <= "2015-12-07" & elapsed > 1800').sort('response_date')

data.groupby('response_date').size().iloc[-20:]



