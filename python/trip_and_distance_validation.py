import os
import pandas as pd
os.chdir('..')
import ttools #homemade module
import gtfs #homemade module
os.chdir('/gpfs2/projects/project-bus_capstone_2016/workspace/share')

# get the sample of parsed AVL data.  Beware, large files take more time.
bustime = pd.read_csv('spark_parse/1203.txt',header=None)

# beware!  Bonan is still working on organizing the extract files.  these columns may change.
bustime.columns = ['ROUTE_ID','latitude','longitude','recorded_time','vehicle_id','TRIP_ID','trip_date','SHAPE_ID',
                   'STOP_ID','distance_stop','distance_shape','status']

bustime.drop_duplicates(['vehicle_id','recorded_time'],inplace=True)
bustime['TRIP_ID'] = bustime['TRIP_ID'].str.replace('MTA NYCT_','')
bustime['TRIP_ID'] = bustime['TRIP_ID'].str.replace('MTABC_','')
bustime.set_index(['ROUTE_ID','TRIP_ID','trip_date','vehicle_id'],inplace=True,drop=True)

# for demonstration, use a subset. Just get data for one trip-date.
tripDateLookup = "2015-12-03" # this is a non-holiday Thursday
bustime = bustime.xs((tripDateLookup),level=(2),drop_level=False)
bustime.sort_index(inplace=True)
bustime['recorded_time'] = bustime['recorded_time'].apply(ttools.parseActualTime,tdate='2015-12-03')
print 'Finished loading BusTime data and and slicing one day.'

bustime['distance_shape'] = bustime['distance_shape'].convert_objects(convert_numeric=True)
bustime['distance_stop'] = bustime['distance_stop'].convert_objects(convert_numeric=True)
bustime['veh_dist_along_shape'] = bustime['distance_shape'] - bustime['distance_stop']

print (bustime['veh_dist_along_shape'].min(),bustime['veh_dist_along_shape'].max())

# create a GroupBy object for convenience, since most analysis is on trip and vehicle
grouped = bustime.groupby(level=(0,1,2,3))
trip_veh_validation = pd.DataFrame(grouped.size(),columns=['N'])
trip_veh_validation['time_range'] = grouped['recorded_time'].max()-grouped['recorded_time'].min()
trip_veh_validation['dist_range'] = grouped['veh_dist_along_shape'].max()-grouped['veh_dist_along_shape'].min()
trip_veh_validation.head(25)

route_dist_grouped = bustime.reset_index().groupby(['ROUTE_ID','STOP_ID','distance_shape']).size()
route_dist_grouped.head(50)

shape_dist_grouped = bustime.groupby(['SHAPE_ID','STOP_ID','distance_shape']).size()
shape_dist_grouped.head(50)

shape_dist_dupes = shape_dist_grouped.groupby(level=(0,1)).size()
shape_dist_dupes[shape_dist_dupes>1]
shape_dist_dupes.name = 'duplicate_count'

dupe_summary = pd.DataFrame(shape_dist_dupes[shape_dist_dupes>1]).merge(shape_dist_grouped.reset_index(level=2),left_index=True,right_index=True,how='left')
dupe_summary.rename(columns={0:'record_count'},inplace=True)
dupe_summary



