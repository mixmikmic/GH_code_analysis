import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import zipfile

get_ipython().run_line_magic('matplotlib', 'inline')

sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
sns.set_context("talk")
sns.set_palette('Set2', 10)

h5_file_path = 'data/gtfs_store.h5'
store = pd.HDFStore(h5_file_path)
local_zip_path = 'data/sample/gtfs.zip' 

with zipfile.ZipFile(local_zip_path) as zf:
    for fn in zf.namelist():
        short = fn.split('.')[0]
        if short not in store:
            store[short] = pd.read_csv(zf.open(fn))
# got this list from running  s.parent_station.value_counts().head(10).index in the loops notebook
store.stops[store.stops.stop_id.isin(['26652', '12961', '10758', '21658', '14679', '3871', '17592', '37412',
       '4967', '15437'])]

import partridge as ptg

#service_ids_by_date = ptg.read_service_ids_by_date(local_zip_path)
#service_ids = service_ids_by_date[datetime.date(2017, 12, 21)]

feed = ptg.feed(local_zip_path, view={
    'trips.txt': {
        'route_id': 13429,
    },
})

t = feed.trips
c = feed.calendar
r = feed.routes

t.groupby('service_id').size()


feed = ptg.feed(local_zip_path, view={
    'trips.txt': {
        'service_id': service_ids,
        'stop_id': ['13089', '13091', '13540'],
    },
})

len(feed.stops)

feed.stops.groupby('zone_id').size().sort_values(ascending=False)

feed.stops[feed.stops.stop_name.str.startswith('מגדל שלום')]

f = feed.stop_times
trip_ids = f[f.stop_id.isin(['13089', '13091', '13540'])].trip_id.unique()
route_ids = feed.trips[feed.trips.trip_id.isin(trip_ids)].route_id.unique()
route_names = feed.routes[feed.routes.route_id.isin(route_ids)].route_short_name.unique()
stop_ids = feed.stop_times[feed.stop_times.trip_id.isin(trip_ids)].stop_id
stop_counts = stop_ids.value_counts()

stops = feed.stops[feed.stops.stop_id.isin(stop_ids.unique())].set_index('stop_id')

route_names

stop_counts.head()

stops = pd.concat((stops, stop_counts), axis=1)

['13089', '13091', '13540']
stops.loc['13089','zone_id'] = 0
stops.loc['13091','zone_id'] = 0
stops.loc['13540','zone_id'] = 0

# change zone for the Shalom Tower stations
# Create scatterplot of dataframe
sns.lmplot('stop_lon', # Horizontal axis
           'stop_lat', # Vertical axis
           data=stops, # Data source
           fit_reg=False, # Don't fix a regression line
           hue="zone_id", # Set color
           scatter_kws={"marker": "D", # Set marker style
                        "s": 100}) # S marker size

# Set title
plt.title('Stations that connect to/from Shalom Tower')

# Set x-axis label
plt.xlabel('lon')

# Set y-axis label
plt.ylabel('lat')



























