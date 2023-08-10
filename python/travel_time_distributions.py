import os
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# these two modules are homemade
os.chdir('..')
import gtfs
import ttools
os.chdir('/gpfs2/projects/project-bus_capstone_2016/workspace/share')

stop_times, tz_sched = gtfs.load_stop_times('2015-12-03','gtfs/')
interpolated = pd.read_csv('dec2015_interpolated.csv')
interpolated = interpolated.merge(stop_times[['arrival_time','stop_sequence']],how='left',left_on=['TRIP_ID','STOP_ID'],right_index=True)
del stop_times, tz_sched # to free up memory
interpolated.set_index(['ROUTE_ID','TRIP_ID','trip_date','vehicle_id','stop_sequence','STOP_ID'],inplace=True)
print 'Finished loading GTFS data and interpolated stop times and merging.'

interpolated['interpolated_arrival_time'] = pd.to_timedelta(interpolated['interpolated_arrival_time'])
interpolated['arrival_time'] = pd.to_timedelta(interpolated['arrival_time'])

grouped = interpolated.groupby(level=(0,1,2,3))
begins = grouped.min()
ends = grouped.max()
travel_times = begins.join(ends,lsuffix='_begin',rsuffix='_end')
def earliest_stop(g):
    return g.idxmin()[-1][-1]
def latest_stop(g):
    return g.idxmax()[-1][-1]
travel_times['earliest_stop'] = grouped.apply(earliest_stop)
travel_times['latest_stop'] = grouped.apply(latest_stop)

travel_times['ttime_schedule'] = (travel_times['arrival_time_end'] - travel_times['arrival_time_begin'])/ttools.datetime.timedelta(seconds=1)
travel_times['ttime_actual'] = (travel_times['interpolated_arrival_time_end'] - travel_times['interpolated_arrival_time_begin'])/ttools.datetime.timedelta(seconds=1)
travel_times['ttime_var'] = travel_times['ttime_actual'] - travel_times['ttime_schedule'] 
# show final dataframe of prepared data
travel_times.head(25)

def peak_hour(x):
    if x > ttools.datetime.timedelta(hours=6) and x < ttools.datetime.timedelta(hours=9):
        return True
    else:
        if x > ttools.datetime.timedelta(hours=16) and x < ttools.datetime.timedelta(hours=19):
            return True
        else:
            return False
travel_times['P_hour'] = map(lambda x:peak_hour(x),travel_times['arrival_time_begin'])

travel_times['ttime_var'].hist(range=(-500,1500),bins=100)

# show without zero-value
travel_times.query('ttime_schedule > 0')['ttime_var'].hist(range=(-500,1500),bins=100)

ttime_var_pct = travel_times.query('ttime_schedule > 0')['ttime_var']/travel_times.query('ttime_schedule > 0')['ttime_schedule']

plt.ttime_var_pct.hist(range=(-0.4,1),bins=100)

fig, ax = plt.subplots(1,1,figsize=(15,8))
ax.hist(ttime_var_pct.values, range=(-0.4,1),bins=100, facecolor='green', alpha=0.4)
plt.ylabel('Number of Occurences', fontsize=14)
plt.xlabel('Running Time Adherence', fontsize=14)
vals = ax.get_xticks()
ax.set_xticklabels(['{:2.1f}%'.format(x*100) for x in vals])
plt.savefig('/gpfs2/projects/project-bus_capstone_2016/workspace//mu529/Bus-Capstone/plots/running_time_adherence.png')
plt.show()

sample_sizes = travel_times.reset_index().groupby(['ROUTE_ID','earliest_stop','latest_stop']).size()
sample_sizes.groupby(level=0).idxmax()

travel_times.loc['B38'].query('earliest_stop == 307460 & latest_stop == 503884')['ttime_schedule'].hist(range=(2000,3500),bins=30)

travel_times.loc['B38'].query('earliest_stop == 307460 & latest_stop == 503884')['ttime_actual'].hist(range=(2000,3500),bins=30)

travel_times.loc['B38'].query('earliest_stop == 307460 & latest_stop == 503884')['ttime_schedule'].describe()

travel_times.loc['B38'].query('earliest_stop == 307460 & latest_stop == 503884')['ttime_actual'].describe()

travel_times.dtypes

travel_times.loc['B38'].query('earliest_stop == 307460 & latest_stop == 503884 & P_hour == True')['ttime_actual'].describe()

travel_times.query('ttime_schedule > 0 & P_hour == True')['ttime_var'].hist(range=(-500,1500),bins=40)

travel_times.query('ttime_schedule > 0 & P_hour == True')['ttime_var'].describe()

travel_times.query('ttime_schedule > 0 & P_hour == False')['ttime_var'].describe()

peak_var_pct = travel_times.query('ttime_schedule > 0 & P_hour == True')['ttime_var']/travel_times.query('ttime_schedule > 0 & P_hour == True')['ttime_schedule']
peak_var_pct.hist(range=(-0.4,1),bins=40)

peak_var_pct.describe()

offpeak_var_pct = travel_times.query('ttime_schedule > 0 & P_hour == False')['ttime_var']/travel_times.query('ttime_schedule > 0 & P_hour == False')['ttime_schedule']
offpeak_var_pct.hist(range=(-0.4,1),bins=40)

offpeak_var_pct.describe()



