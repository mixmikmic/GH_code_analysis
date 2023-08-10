import os
import pandas as pd
get_ipython().magic('matplotlib inline')
from datetime import timedelta
sample_file = '/gpfs2/projects/project-bus_capstone_2016/workspace/share/20141201_dot.csv'

dot_sample = pd.read_csv(sample_file,header=None,names=['Vehicle','Route','Direction','Phase','NMEA'])

dot_sample.head()

# apply function to extract time element from NMEA string
def time_from_nmea(s):
    h = int(s[7:9])
    m = int(s[9:11])
    s = float(s[11:16])
    return timedelta(hours=h,minutes=m,seconds=s)
dot_sample['timestamp'] = dot_sample.NMEA.apply(time_from_nmea)

dot_sample.sort(['Vehicle','timestamp'],inplace=True)
dot_sample['elapsed'] = dot_sample.groupby(['Vehicle'])['timestamp'].diff()/timedelta(seconds=1)

dot_sample.head(10)

dot_sample['elapsed'].dropna().hist(range=(0,89),bins=45)

# percent of AVL pings received under 35 seconds from previous
sum(dot_sample['elapsed'].dropna()<35.0)/(1.0*(len(dot_sample['elapsed'].dropna())))

# number of unique Vehicle IDs (expect <5700 on any given day, accounting for out-of-service vehicles: http://web.mta.info/nyct/facts/about_us.htm)
len(dot_sample.Vehicle.unique())

# number of pings recorded in 5-minute intervals, all service status
ts_float = (dot_sample.timestamp/timedelta(hours=1))-5
ts_float.hist(bins=288)

# number of pings recorded in 5-minute intervals, IN PROGRESS only
ts_float = (dot_sample.query('Phase == "IN_PROGRESS"').timestamp/timedelta(hours=1))-5
ts_float.hist(bins=288)

# number of IN_PROGRESS pings recorded by Route
dot_sample.query('Phase == "IN_PROGRESS"').groupby('Route').size()



