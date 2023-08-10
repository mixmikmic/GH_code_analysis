import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
import time
import datetime 
get_ipython().magic('matplotlib inline')
import sys
sys.path.append('/gpfs2/projects/project-bus_capstone_2016/workspace/jz2308/bus-Capstone')

# these two modules are homemade
import gtfs
import arrivals
import ttools
os.chdir('/gpfs2/projects/project-bus_capstone_2016/workspace/share')

base = datetime.date(2015, 1, 1)
schedule_samples = [str(base + datetime.timedelta(days=x)) for x in range(0, 365)]
#schedule_samples = ['2015-01-04','2015-04-05','2015-06-27','2015-07-06','2015-09-05','2015-09-15','2015-10-12']
collected = pd.DataFrame()
for sdate in schedule_samples[0:1]:
    stops = gtfs.load_stops(sdate,'gtfs/')
    stops['schedule_sample_date'] = sdate
    collected = collected.append(stops[['stop_lat','stop_lon','stop_name','schedule_sample_date']].drop_duplicates())

collected.reset_index().head()

collected.reset_index().head()

# fixed column name, for proper indexing later
reliability_metrics = pd.read_csv('performance.csv',names = ["route_id", "stop_id", "schedule_sample_date", "ontime_ratio", "peak_hours wait ass","offpeak hours wait ass"])

reliability_metrics.query('route_id == "M15"')

reliability_metrics['peak_hours wait ass'].replace('None', None, inplace=True)

pd.DataFrame(reliability_metrics.reset_index().groupby(['stop_id']).mean().reset_index()).head()

stops_metrics = collected.reset_index().merge(pd.DataFrame(reliability_metrics.reset_index().groupby('stop_id').mean().reset_index()))

stops_metrics.head()

stops_metrics = stops_metrics.groupby('stop_id').mean()[['stop_lat','stop_lon','ontime_ratio','offpeak hours wait ass']]

stops_metrics.head()

stops_metrics.to_csv('metrics-by-stop-2015.csv')

# get all the schedule data. (subset can be created later)
trips = gtfs.load_trips('2015-12-03','gtfs/')
# stops = gtfs.load_stops('2015-12-03','gtfs/')
stop_times, tz_sched = gtfs.load_stop_times('2015-12-03','gtfs/')
print 'Finished loading GTFS data.'
joined = stop_times.reset_index(level=1)[['stop_id','stop_sequence']].join(trips[['route_id','direction_id']])
unique_stops = joined.reset_index().drop_duplicates(subset=['route_id','direction_id','stop_id','stop_sequence']).drop('trip_id',axis=1)
unique_stops = unique_stops.set_index(['route_id','stop_id'],verify_integrity=False)
dupe_inds = unique_stops.index[unique_stops.index.duplicated()]
unique_stops = unique_stops.loc[dupe_inds]

unique_stops.loc['M15']

collected_reset = collected.reset_index()

unique_stops_reset = unique_stops.reset_index()

collected_reset.head()

unique_stops_reset.head()

line_sample = unique_stops_reset.query('route_id == "M15" & direction_id == 0')

line_sample.set_index(['route_id','stop_id'],inplace=True)
line_sample.head()

avg_offpeak = reliability_metrics.groupby(['route_id','stop_id'])['offpeak hours wait ass'].mean()

avg_offpeak.head()

line_sample.loc[:,'avg_offpeak'] = avg_offpeak

line_sample

line_sample.join(collected[['stop_lat','stop_lon']],how='left').head()

def get_sample(route_id,direction_id):
    line_sample = unique_stops_reset.query('route_id == @route_id & direction_id == @direction_id')
    line_sample.set_index(['route_id','stop_id'],inplace=True)
    avg_offpeak = reliability_metrics.groupby(['route_id','stop_id'])['offpeak hours wait ass'].mean()
    line_sample.loc[:,'avg_offpeak'] = avg_offpeak
    return line_sample.join(collected[['stop_lat','stop_lon']],how='left')

# demonstrate
get_sample('M15',0)

line_metrics.query("Line=='B1' and direction_id==1").to_csv('metrics-by-line-stop-B1-1-2015.csv')

line_metrics.query("Line=='B1' and direction_id==0").to_csv('metrics-by-line-stop-B1-0-2015.csv')

line_metrics.query("Line=='M5' and direction_id==1").to_csv('metrics-by-line-stop-M5-1-2015.csv')

line_metrics.query("Line=='M5' and direction_id==0").to_csv('metrics-by-line-stop-M5-0-2015.csv')

line_metrics.query("Line=='M15' and direction_id==0").to_csv('metrics-by-line-stop-M15-0-2015.csv')

line_metrics.query("Line=='M15' and direction_id==1").to_csv('metrics-by-line-stop-M15-1-2015.csv')

import matplotlib.pyplot as plt
plt.figure(figsize=(40,20))
plt.scatter(stops_metrics.stop_lon,stops_metrics.stop_lat,c=stops_metrics.ontime_ratio,s=55)

