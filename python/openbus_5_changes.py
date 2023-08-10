import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

get_ipython().run_line_magic('matplotlib', 'inline')

2+2















from gtfs_utils import *

conn = ftp_connect()
ftp_dir = get_ftp_dir(conn)
UPTODATE = 90 #days
our_uptodateness = get_uptodateness(ftp_dir)

if our_uptodateness > UPTODATE:
    get_ftp_file(conn)
    get_ftp_file(conn, file_name = 'Tariff.zip', local_zip_path = 'data/sample/tariff.zip' )

conn.quit()

tariff_df = extract_tariff_df(local_zip_path = 'data/sample/tariff.zip')
south = [
    {
        'zone_name': 'מצפה רמון',
        'zone_id': '903'
    },
    {
        'zone_name': 'ערבה',
        'zone_id': '902'
    },
    {
        'zone_name': 'אילת',
        'zone_id': '901'
    },]
south = pd.DataFrame(south)
tariff_df = tariff_df.append(south)

LOCAL_ZIP_PATH = 'data/sample/gtfs.zip' 

import partridge as ptg

service_ids_by_date = ptg.read_service_ids_by_date(LOCAL_ZIP_PATH)
service_ids = service_ids_by_date[datetime.date(2017, 12, 21)]

feed = ptg.feed(LOCAL_ZIP_PATH, view={
    'trips.txt': {
        'service_id': service_ids,
    },
})

def to_timedelta(df):
    '''
    Turn time columns into timedelta dtype
    '''
    cols = ['arrival_time', 'departure_time']
    numeric = df[cols].apply(pd.to_timedelta, unit='s')
    df = df.copy()
    df[cols] = numeric
    return df

s = feed.stops
r = feed.routes
t = (feed.trips
     .assign(route_id=lambda x: pd.Categorical(x['route_id'])))

f = (feed.stop_times[['trip_id', 'departure_time', 'arrival_time', 'stop_id', 'stop_sequence']]
     .assign(date = datetime.date(2017, 12, 21))
     .merge(s[['stop_id', 'stop_name', 'stop_lat', 'stop_lon', 'zone_id']], on='stop_id')
     # Much faster joins and slices with Categorical dtypes
     .merge(tariff_df[['zone_id', 'zone_name']], on='zone_id')
     .assign(zone_id=lambda x: pd.Categorical(x['zone_id']))
     .assign(zone_name=lambda x: pd.Categorical(x['zone_name']))
     .merge(t[['trip_id', 'route_id', 'direction_id']], on='trip_id')
     .merge(r[['route_id', 'route_short_name', 'route_long_name']], on='route_id')
     .assign(route_id=lambda x: pd.Categorical(x['route_id']))
     .pipe(to_timedelta)
    )
f.head()

import pickle
import os

# using a custom groupby aggregation function, this is VERY inefficient,
# but far more readable than doing some other magic, so I'm sticking to it for now

def apply_and_pickle(f, pkl_path, apply_func):
    if not os.path.exists(pkl_path):
        ret_df = f.groupby('trip_id').apply(apply_func)
        # since it's so inefficient I pickle it
        with open(pkl_path, 'wb') as pkl_file:
            pickle.dump(orig_dest_only, pkl_file)
    else:
        with open(pkl_path, 'rb') as pkl_file:
            ret_df = pickle.load(pkl_file)
        return ret_df

def first_last(df):
    return df.sort_values(by='stop_sequence').iloc[[0, -1]]

orig_dest_only = apply_and_pickle(f, 'softloop.pkl', first_last)

fig, ax = plt.subplots(figsize=(10,7))
ax.set_yticks(np.arange(32)*50)
orig_dest_only[orig_dest_only.stop_sequence!=1].groupby('stop_id').size().sort_values(ascending=False).hist(bins=147, ax=ax)

orig_dest_only[orig_dest_only.stop_sequence!=1].groupby('stop_id').size().sum()

t.shape

f[f.stop_sequence==1].shape

f[f.stop_sequence==2].shape

idx = f.groupby(['trip_id'], sort=False)['arrival_time'].transform(max) == f['arrival_time']
f[idx].sort_values(by='trip_id').head()

f[idx].shape

f[idx].trip_id.value_counts().head()

st = feed.stop_times
st[((st.trip_id=='24167955_171217') & (st.stop_sequence==31)) ]

f[((f.trip_id=='24167955_171217') & (f.stop_sequence==31)) ]

ftest = (feed.stop_times[['trip_id', 'departure_time', 'arrival_time', 'stop_id', 'stop_sequence']]
     .assign(date = datetime.date(2017, 12, 21))
     .merge(s[['stop_id', 'stop_name', 'stop_lat', 'stop_lon', 'zone_id']], on='stop_id')
      # Much faster joins and slices with Categorical dtypes
     .merge(tariff_df[['zone_id', 'zone_name']], on='zone_id')

    )
ftest[ ((ftest.trip_id=='10230038_171217') & (ftest.stop_sequence==31)) ]

tariff_df[tariff_df.zone_id=='232']

tariff_df.groupby(['zone_id', 'zone_name']).size()

ftry = (feed.stop_times[['trip_id', 'departure_time', 'arrival_time', 'stop_id', 'stop_sequence']]
     .assign(date = datetime.date(2017, 12, 21))
     .merge(s[['stop_id', 'stop_name', 'stop_lat', 'stop_lon', 'zone_id']], on='stop_id')
      # Much faster joins and slices with Categorical dtypes
     .merge(tariff_df.groupby(['zone_id', 'zone_name']).size().reset_index()[['zone_id', 'zone_name']], on='zone_id')
    )
ftry [ ((ftry.trip_id=='10230038_171217') & (ftry.stop_sequence==31)) ]

new_f = (feed.stop_times[['trip_id', 'departure_time', 'arrival_time', 'stop_id', 'stop_sequence']]
     .assign(date = datetime.date(2017, 12, 21))
     .merge(s[['stop_id', 'stop_name', 'stop_lat', 'stop_lon', 'zone_id']], on='stop_id')
     # Much faster joins and slices with Categorical dtypes
     .merge(tariff_df.groupby(['zone_id', 'zone_name']).size().reset_index()[['zone_id', 'zone_name']], on='zone_id')
     .assign(zone_id=lambda x: pd.Categorical(x['zone_id']))
     .assign(zone_name=lambda x: pd.Categorical(x['zone_name']))
     .merge(t[['trip_id', 'route_id', 'direction_id']], on='trip_id')
     .merge(r[['route_id', 'route_short_name', 'route_long_name']], on='route_id')
     .assign(route_id=lambda x: pd.Categorical(x['route_id']))
     .pipe(to_timedelta)
    )
new_f.head()

idx = new_f.groupby(['trip_id'], sort=False)['arrival_time'].transform(max) == new_f['arrival_time']
new_f[idx].trip_id.value_counts().head()

new_f[idx].shape

t = t.set_index('trip_id')
missing_trips = t[~t.index.isin(new_f.trip_id)]
missing_trips

st[st.trip_id.isin(missing_trips.index)].shape

test_trips = t.head()
test_trips

st[st.trip_id.isin(test_trips.index)].shape

