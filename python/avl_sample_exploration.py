import os
import pandas as pd
import numpy as np
import sys
import pylab as pl
sys.path.append('/gpfs2/projects/project-bus_capstone_2016/workspace/mu529/Bus-Capstone')
import ttools
os.chdir('/gpfs2/projects/project-bus_capstone_2016/workspace/share')
get_ipython().magic('matplotlib inline')

# import sample and slice one day
avl_data = pd.read_csv('newdata_parsed.csv')
oneday = avl_data.query('TripDate=="2016-06-13"')
del avl_data

# organize the data: sort, drop duplicates, and parse vehicle timestamps
oneday = oneday.sort(columns=['vehicleID','RecordedAtTime'])
oneday.drop_duplicates(subset=['vehicleID','RecordedAtTime','Latitude','Longitude'],inplace=True)
oneday['RecordedAtTime_parsed'] = oneday.RecordedAtTime.apply(ttools.parseActualTime,tdate='2016-06-13')
# Also add a column showing the time since the previous ping from the same vehicle.
oneday['TimeSinceLastPing'] = oneday.RecordedAtTime_parsed.diff()

# check for data coming from two vehicles on the same trip
oneday.groupby(['Line','Trip','TripDate','vehicleID']).size()

responsetimes = oneday.ResponseTimeStamp.unique()
responsetimes.sort()
responsetimes = pd.Series(data=responsetimes,index=responsetimes)
responsetimes = responsetimes.apply(ttools.parseActualTime,tdate='2016-06-13')
responsetimediffs = pd.DataFrame(responsetimes.diff(),columns=['TimeSinceLastSiriResponse'])

responsetimediffs.head()

oneday = oneday.merge(responsetimediffs,how='left',left_on='ResponseTimeStamp',right_index=True)

# gap for any reason
gaps = oneday.TimeSinceLastPing>ttools.datetime.timedelta(seconds=33)
oneday['gap_ind'] = gaps
gaps.head()

# percent of all pings that follow a gap (that is, longer than 33 seconds since the previous ping)
sum(gaps)/(len(gaps)*1.0)

# these are definitely obscured by Siri gap, because the gap since the last siri response is even longer.
oneday['siri_gap_ind'] = False
oneday.loc[gaps,'siri_gap_ind'] = oneday.TimeSinceLastPing[gaps] < oneday.TimeSinceLastSiriResponse[gaps]
# report percentage
sum(oneday['siri_gap_ind'])/(len(gaps)*1.0)

# these are gaps for which the last Siri response was at least 33 seconds after the last ping, so there
# should have been at least one more ping recorded.  These are definitely a vehicle gap (at least, partially)
oneday['veh_gap_ind'] = False
oneday.loc[gaps,'veh_gap_ind'] = oneday.TimeSinceLastSiriResponse[gaps] < (oneday.TimeSinceLastPing[gaps] - ttools.datetime.timedelta(seconds=33))
# need to exclude cases where the last Siri response was less than 30 seconds ago, since that would result in duplicate reports.
oneday.loc[oneday.TimeSinceLastSiriResponse < ttools.datetime.timedelta(seconds=30),'veh_gap_ind'] = False
# report percentage
sum(oneday['veh_gap_ind'])/(len(gaps)*1.0)

# gap rates for any line, by line
gap_rates = oneday.groupby('Line')['gap_ind'].sum()/oneday.groupby('Line').size()
gap_rates.hist(range=(0.3,0.6),bins=30,color='cyan')
pl.xlabel("Gap ratio by line",fontsize=14)
pl.ylabel("Frequency",fontsize=14)

# veh-caused gap rates by vehicle
vehicle_gap_rates = oneday.groupby('vehicleID')['veh_gap_ind'].sum()/oneday.groupby('vehicleID').size()
vehicle_gap_rates.hist(range=(0.0,0.05),bins=40,color='cyan')
pl.xlabel("Gap ratio by vehicle",fontsize=14)
pl.ylabel("Frequency",fontsize=14)

oneday['response_hour'] = oneday.ResponseTimeStamp.str[11:13]
hourly_gap_rates = oneday.groupby('response_hour')['gap_ind'].sum()/oneday.groupby('response_hour').size()
hourly_gap_rates.plot(color='black')
pl.xlabel("Hour of the day",fontsize=14,size=15)
pl.ylabel("Gap ratio",fontsize=14)

