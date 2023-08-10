example_date = '2015-11-01'
gtfs_path = '/gpfs2/projects/project-bus_capstone_2016/workspace/share/gtfs/'
code_filepath = '..'
avl_sample = '/gpfs2/projects/project-bus_capstone_2016/workspace/share/old_junk/20151101_parsed.csv'

import os
import pandas as pd
import numpy as np
get_ipython().magic('matplotlib inline')

# these two modules are homemade
os.chdir(code_filepath)
import ttools #homemade module
import gtfs #homemade module

# get all the schedule data. (subset can be created later)
trips = gtfs.load_trips(example_date,gtfs_path)
stops = gtfs.load_stops(example_date,gtfs_path)
stop_times, tz_sched = gtfs.load_stop_times(example_date,gtfs_path)
tcal = gtfs.TransitCalendar(example_date,gtfs_path)
print 'Finished loading GTFS data.'

# get the sample of parsed AVL data.  Beware, large files take more time.
bustime = pd.read_csv(avl_sample)

bustime.columns = ['vehicle_id','route','timestamp','lat','lon','trip_id','trip_date','shape_id',
                   'next_stop_id','est_arrival','dist_from_stop','stop_dist_on_trip','presentable_dist','response']
bustime.drop_duplicates(['vehicle_id','timestamp'],inplace=True)
bustime['trip_id'] = bustime['trip_id'].str.replace('MTA NYCT_','')
bustime['trip_id'] = bustime['trip_id'].str.replace('MTABC_','')
bustime.set_index(['route','trip_id','trip_date','vehicle_id'],inplace=True,drop=True)

bustime.groupby(level=2).size()

# for demonstration, use a subset. Just get data for one trip-date.
tripDateLookup = "2015-11-01" # this is a non-holiday Monday
bustime = bustime.xs((tripDateLookup),level=(2),drop_level=False)
bustime.sort_index(inplace=True)
print 'Finished loading BusTime data and and slicing one day.'

# Filter the service_ids for those applicable to this date
# gtfs.TransitCalendar class now correctly adjusts for exception dates
active_services = tcal.get_service_ids('2015-11-01') 
# Load a sepate trips dataframe but use only one index level
trips_ = gtfs.load_trips(example_date,gtfs_path).reset_index().set_index(['service_id'])
# Generate list of all trip_ids that are scheduled for those service_ids
gtfs_trip_ids = trips_.loc[active_services]['trip_id'].unique()
# Generate list of all trip_ids in BusTime subset
bustime_trip_ids = bustime.index.get_level_values(1).unique()

len(gtfs_trip_ids)

len(bustime_trip_ids)

# parse times into numeric
ts_parsed = bustime['timestamp'].apply(ttools.parseActualTime,tdate='2015-11-01')

(ts_parsed/ttools.datetime.timedelta(hours=1)).hist(range=(0,30),bins=180)

set1 = set(gtfs_trip_ids)
set2 = set(bustime_trip_ids)

unmatched = set1.symmetric_difference(set2) # goes both ways
len(unmatched)

pct_missing_by_route = trips.loc[unmatched].groupby('route_id').size()/trips.groupby('route_id').size()
pct_missing_by_route.sort(ascending=False)
pct_missing_by_route

stop_hour = stop_times.reset_index().set_index('trip_id')['arrival_time'].apply(ttools.parseTime)/ttools.datetime.timedelta(hours=1)
stop_hour.loc[unmatched].hist(bins=48)

# Distribution of trip durations - ALL scheduled trips
pd.Series(stop_hour.groupby(level=0).max() - stop_hour.groupby(level=0).min()).hist(bins=40)

# Distribution of trip durations - MISSING trips only
pd.Series(stop_hour.loc[unmatched].groupby(level=0).max() - stop_hour.loc[unmatched].groupby(level=0).min()).hist(bins=40)

gtfs_trip_ids[16130] = ''
str_list = filter(None, gtfs_trip_ids)

len(str_list)

gtfs_trip_ids = np.asarray(str_list)

start_bins = (10*stop_hour.loc[gtfs_trip_ids].groupby(level=0).min()).apply(np.floor).dropna().astype(int)
end_bins = (10*stop_hour.loc[gtfs_trip_ids].groupby(level=0).max()).apply(np.floor).dropna().astype(int)

# time is divided into bins of 0.1 hours (6 minutes)
# make a 2D binary matrix indicating when the trip is active during each time bin.
time_bin_matrix = np.zeros((len(start_bins),max(end_bins)+1))
counter = 0
for i,v_start in start_bins.iteritems():
    v_end = end_bins.loc[i]
    time_bin_matrix[counter,v_start:v_end] = 1
    counter += 1

start_bins_bustime = (10*(ts_parsed.groupby(level=(1,3)).min()/ttools.datetime.timedelta(hours=1))).apply(np.floor).astype(int)
end_bins_bustime = (10*(ts_parsed.groupby(level=(1,3)).max()/ttools.datetime.timedelta(hours=1))).apply(np.floor).astype(int)
time_bin_matrix_bustime = np.zeros((len(start_bins_bustime),max(end_bins_bustime)+1))
counter = 0
for i,v_start in start_bins_bustime.iteritems():
    v_end = end_bins_bustime.loc[i]
    time_bin_matrix_bustime[counter,v_start:v_end] = 1
    counter += 1

pd.DataFrame(time_bin_matrix_bustime.sum(axis=0),columns=['BusTime_Vehicles']).join(pd.Series(time_bin_matrix.sum(axis=0),name='GTFS_Vehicles')).plot()



