example_date = '2015-12-03'
gtfs_path = '/gpfs2/projects/project-bus_capstone_2016/workspace/share/gtfs/'
code_filepath = '..'

import os
import pandas as pd
import numpy as np
get_ipython().magic('matplotlib inline')

# these modules are homemade
os.chdir(code_filepath)
import ttools
import gtfs

# get all the schedule data. (subset can be created later)
trips = gtfs.load_trips(example_date,gtfs_path)
stops = gtfs.load_stops(example_date,gtfs_path)
stop_times, tz_sched = gtfs.load_stop_times(example_date,gtfs_path)
tcal=gtfs.TransitCalendar(example_date,gtfs_path)

active_services = tcal.get_service_ids(example_date)
active_trips = trips.service_id.isin(active_services)
active_stop_times = stop_times.reset_index().set_index('trip_id').loc[active_trips]
print 'Finished loading GTFS data.'

active_stop_times.set_index('stop_id',append=True,inplace=True)

active_stop_times.groupby(level=0).size().hist(range=(0,60),bins=30)

print len(active_stop_times)
print sum(active_trips)
print 1.0*len(active_stop_times)/sum(active_trips)

active_stop_times['arrival_time'] = pd.to_timedelta(active_stop_times['arrival_time'])
active_stop_times['departure_time'] = pd.to_timedelta(active_stop_times['departure_time'])

trip_durations = active_stop_times.groupby(level=(0))['arrival_time'].max()- active_stop_times.groupby(level=(0))['arrival_time'].min()

trip_durations.describe()

(trip_durations/ttools.datetime.timedelta(minutes=1)).hist(bins=40)

list(trip_durations[trip_durations < ttools.datetime.timedelta(minutes=5)].index)

trips.loc['GA_D5-Weekday-SDon-118800_B60_728']

stop_times.loc['GA_D5-Weekday-SDon-118800_B60_728']

