import pandas as pd

url = "https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2016-01.csv"

df = pd.read_csv(url)

df.shape

df[['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'fare_amount']].head(10)

def parseTrips(df):
    
    tripTable = pd.DataFrame()
    tripTable['start_time'] = pd.to_datetime(df['tpep_pickup_datetime'])
    tripTable['end_time'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    tripTable['passenger_count'] = df['passenger_count']
    tripTable['start_lat'] = df['pickup_latitude']
    tripTable['start_lon'] = df['pickup_longitude']
    tripTable['end_lat'] = df['dropoff_latitude']
    tripTable['end_lon'] = df['dropoff_longitude']
    tripTable['duration'] = tripTable['end_time'] - tripTable['start_time']
    tripTable['duration'] = tripTable['duration'].dt.total_seconds()
    tripTable['vehicle_type'] = "yellow"
    tripTable = tripTable[((tripTable['start_lat'] != 0) & (tripTable['end_lat'] != 0))]
    
    tripTable = tripTable.sort_values(by=['start_time', 'end_time'])
    return tripTable

tripTable = parseTrips(df)

tripTable.head()

tripTable.tail()

from bokeh.plotting import figure, output_notebook, show
output_notebook()

import pandas as pd
import datashader as ds
import datashader.transfer_functions as tf
from functools import partial
from datashader.utils import export_image
from datashader.colors import colormap_select, Greys9, Hot, viridis, inferno

def clipDataToBoundingBox(df, bbox):
    min_lon, min_lat, max_lon, max_lat = bbox
    clipped = df[((df['start_lat'] >= min_lat) & (df['start_lat'] <= max_lat) &                  (df['start_lon'] >= min_lon) & (df['start_lon'] <= max_lon) &                  (df['end_lat'] >= min_lat)   & (df['end_lat']   <= max_lat) &                  (df['end_lon'] >= min_lon)   & (df['end_lon']   <= max_lon))]
    return clipped

nyc_bbox = [-74.278564,40.485604,-73.609772,40.945676]
tripTable_clipped = clipDataToBoundingBox(tripTable, nyc_bbox)

background = "black"
export = partial(export_image, background = background)
cm = partial(colormap_select, reverse=(background!="black"))

cvs = ds.Canvas(plot_width=1000, plot_height=1000)
agg = cvs.points(tripTable_clipped, 'start_lon', 'start_lat')
export(tf.shade(agg, cmap = cm(Hot,0.2), how='eq_hist'),"passenger_count")

lga_bbox = [-73.893785,40.758245,-73.859269,40.782900]
lga_pickups = clipDataToBoundingBox(tripTable, lga_bbox)

get_ipython().run_line_magic('pinfo', 'tf.shade')

cvs = ds.Canvas(plot_width=800, plot_height=600)
agg = cvs.points(lga_pickups, 'start_lon', 'start_lat')
export(tf.shade(agg, cmap = cm(Hot,0.3), how='eq_hist'),"passenger_count")

url = "https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2016-02.csv"

df = pd.read_csv(url)

df.shape

df.head()

tripTable = parseTrips(df)

tripTable.head()

nyc_bbox = [-74.278564,40.485604,-73.609772,40.945676]
tripTable_clipped = clipDataToBoundingBox(tripTable, nyc_bbox)

background = "black"
export = partial(export_image, background = background)
cm = partial(colormap_select, reverse=(background!="black"))

cvs = ds.Canvas(plot_width=1000, plot_height=1000)
agg = cvs.points(tripTable_clipped, 'start_lon', 'start_lat')
export(tf.shade(agg, cmap = cm(Hot,0.2), how='eq_hist'),"passenger_count")

url = "https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2016-03.csv"

df = pd.read_csv(url)
tripTable = parseTrips(df)

nyc_bbox = [-74.278564,40.485604,-73.609772,40.945676]
tripTable_clipped = clipDataToBoundingBox(tripTable, nyc_bbox)

background = "black"
export = partial(export_image, background = background)
cm = partial(colormap_select, reverse=(background!="black"))

cvs = ds.Canvas(plot_width=1000, plot_height=1000)
agg = cvs.points(tripTable_clipped, 'start_lon', 'start_lat')
export(tf.shade(agg, cmap = cm(Hot,0.2), how='eq_hist'),"passenger_count")

url = "https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2016-04.csv"

df = pd.read_csv(url)
tripTable = parseTrips(df)

nyc_bbox = [-74.278564,40.485604,-73.609772,40.945676]
tripTable_clipped = clipDataToBoundingBox(tripTable, nyc_bbox)

background = "black"
export = partial(export_image, background = background)
cm = partial(colormap_select, reverse=(background!="black"))

cvs = ds.Canvas(plot_width=1000, plot_height=1000)
agg = cvs.points(tripTable_clipped, 'start_lon', 'start_lat')
export(tf.shade(agg, cmap = cm(Hot,0.2), how='eq_hist'),"passenger_count")

url = "https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2016-05.csv"

df = pd.read_csv(url)
tripTable = parseTrips(df)

nyc_bbox = [-74.278564,40.485604,-73.609772,40.945676]
tripTable_clipped = clipDataToBoundingBox(tripTable, nyc_bbox)

background = "black"
export = partial(export_image, background = background)
cm = partial(colormap_select, reverse=(background!="black"))

cvs = ds.Canvas(plot_width=1000, plot_height=1000)
agg = cvs.points(tripTable_clipped, 'start_lon', 'start_lat')
export(tf.shade(agg, cmap = cm(Hot,0.2), how='eq_hist'),"passenger_count")



