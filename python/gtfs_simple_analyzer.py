import pandas as pd
import numpy as np
import urllib.request


# Download the file from `url` and save it locally under `gtfs.zip`, then extract:
def gtfs_downloader(url):
    file_name = 'gtfs.zip'
    urllib.request.urlretrieve(url, file_name)
    import zipfile
    with zipfile.ZipFile(file_name,"r") as zip_ref:
        zip_ref.extractall("gtfs/")

#Give the URL of the TransitFeeds gtfs you want to analyze and create dataframes.
# url = 'http://transitfeeds.com/p/vta/45/20170321/download'
url = 'http://transitfeeds.com/p/vta/45/latest/download'
gtfs_downloader(url)
trips = pd.read_csv('gtfs/trips.txt')
st = pd.read_csv('gtfs/stop_times.txt')
routes = pd.read_csv('gtfs/routes.txt')
stops = pd.read_csv('gtfs/stops.txt')

stops.head()

trips.head()

st.head()

#Merge only a few columns from stop times with trips to see all times that trips hit certain stops
pd.merge(trips,st[['trip_id','stop_id','arrival_time']])

#This is a double merge to give you the stop_name as well.
stops_by_route = pd.merge(pd.merge(trips,st[['trip_id','stop_id','arrival_time']]),stops[['stop_id','stop_code','stop_name']])
stops_by_route

# Only query for weekday trips.
stops_by_route.query("service_id=='Weekdays'").groupby(['route_id','stop_code']).first().reset_index()

# Mark timepoints since arrival_time is null for a non-timepoint, thus where it is not null it is a timepoint.
stops_by_route['timepoint']=~pd.isnull(stops_by_route['arrival_time'])
stops_by_route

#Write to a csv
stops_by_route[['route_id','stop_code','timepoint','stop_name']].to_csv('../qa-qc-realtime/stops_by_route.csv',index=False)

#How many trips there are on weekdays.
trips.query("service_id=='Weekdays'").count()

