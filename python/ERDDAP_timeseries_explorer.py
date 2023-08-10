import numpy as np
import pandas as pd

import pendulum

import ipyleaflet as ipyl
import bqplot as bq
import ipywidgets as ipyw

from erddapy import ERDDAP

endpoint = 'https://erddap-uncabled.oceanobservatories.org/uncabled/erddap'

initial_standard_name = 'sea_water_temperature'

nchar = 8 # number of characters for short dataset name
cdm_data_type = 'Point'
center = [35, -100]
zoom = 1

min_time = '2017-08-01T00:00:00Z'
max_time = '2017-08-03T00:00:00Z'

endpoint = 'http://erddap.sensors.ioos.us/erddap'

initial_standard_name = 'sea_surface_wave_significant_height'

nchar = 9 # number of characters for short dataset name
cdm_data_type = 'TimeSeries'
center = [35, -100]
zoom = 3

now = pendulum.utcnow()
max_time = now
min_time = now.subtract(weeks=2) 

min_time = '2017-11-01T00:00:00Z'
max_time = '2017-11-11T00:00:00Z'

endpoint = 'https://gamone.whoi.edu/erddap'

initial_standard_name = 'sea_water_temperature'

nchar = 9 # number of characters for short dataset name
cdm_data_type = 'TimeSeries'
center = [35, -100]
zoom = 3

now = pendulum.utcnow()
max_time = now
min_time = now.subtract(weeks=2) 

min_time = '2011-05-05T00:00:00Z'
max_time = '2011-05-15T00:00:00Z'

endpoint = 'http://ooivm1.whoi.net/erddap'

initial_standard_name = 'wind_speed'

nchar = 3 # number of characters for short dataset name
cdm_data_type = 'TimeSeries'
center = [42.5, -68]
zoom = 6

now = pendulum.utcnow()
max_time = now
min_time = now.subtract(weeks=2) 

endpoint = 'http://www.neracoos.org/erddap'

initial_standard_name = 'significant_height_of_wind_and_swell_waves'

nchar = 3 # number of characters for short dataset name
cdm_data_type = 'TimeSeries'
center = [42.5, -68]
zoom = 6

now = pendulum.utcnow()
max_time = now
min_time = now.subtract(weeks=2) 

e = ERDDAP(server_url=endpoint)

url='{}/categorize/standard_name/index.csv'.format(endpoint)
df = pd.read_csv(url, skiprows=[1, 2])
vars = df['Category'].values

dpdown = ipyw.Dropdown(options=vars, value=initial_standard_name)

def download_csv(url):
    return pd.read_csv(url, index_col='time', parse_dates=True, skiprows=[1])

def point(dataset, lon, lat, nchar):
    geojsonFeature = {
        "type": "Feature",
        "properties": {
            "datasetID": dataset,
            "short_dataset_name": dataset[:nchar]
        },
        "geometry": {
            "type": "Point",
            "coordinates": [lon, lat]
        }
    };
    geojsonFeature['properties']['style'] = {'color': 'Grey'}
    return geojsonFeature

def adv_search(e, standard_name, cdm_data_type, min_time, max_time):
    try:
        search_url = e.get_search_url(response='csv', cdm_data_type=cdm_data_type.lower(), items_per_page=100000,
                                  standard_name=standard_name, min_time=min_time, max_time=max_time)
        df = pd.read_csv(search_url)
    except:
        df = []
        if len(var)>14:
            v = '{}...'.format(standard_name[:15])
        else:
            v = standard_name
        figure.title = 'No {} found in this time range. Pick another variable.'.format(v)
        figure.marks[0].y = 0.0 * figure.marks[0].y
    return df

def alllonlat(e, cdm_data_type, min_time, max_time):
    url='{}/tabledap/allDatasets.csv?datasetID%2CminLongitude%2CminLatitude&cdm_data_type=%22{}%22&minTime%3C={}&maxTime%3E={}'.format(e.server_url,cdm_data_type,max_time,min_time)
    df = pd.read_csv(url, skiprows=[1])
    return df

def stdname2geojson(e, standard_name, min_time, max_time, nchar):
    '''return geojson containing lon, lat and datasetID for all matching stations'''
    # find matching datsets using Advanced Search
    dfa = adv_search(e, standard_name, cdm_data_type, min_time, max_time)
    if isinstance(dfa, pd.DataFrame):
        datasets = dfa['Dataset ID'].values
        # find lon,lat values from allDatasets 
        dfll = alllonlat(e, cdm_data_type, min_time, max_time)
        # extract lon,lat values of matching datasets from allDatasets dataframe
        dfr = dfll[dfll['datasetID'].isin(dfa['Dataset ID'])]
        # contruct the GeoJSON using fast itertuples
        geojson = {'features':[point(row[1],row[2],row[3],nchar) for row in dfr.itertuples()]}
    else:
        geojson = {'features':[]}
        datasets = []
    return geojson, datasets

def click_handler(event=None, id=None, properties=None):
    datasetID = properties['datasetID']
    kwargs = {'time%3E=': min_time, 'time%3C=': max_time}
    df, var = get_data(datasetID, dpdown.value, kwargs)
    figure.marks[0].x = df.index
    figure.marks[0].y = df[var]
    figure.title = '{} - {}'.format(properties['short_dataset_name'], var)

def update_dpdown(change):
    standard_name = change['new']
    data, datasets = stdname2geojson(e, standard_name, min_time, max_time, nchar)
    feature_layer = ipyl.GeoJSON(data=data)
    feature_layer.on_click(click_handler)
    map.layers = [map.layers[0], feature_layer]

dpdown.observe(update_dpdown, names=['value'])

def get_data(dataset, standard_name, kwargs):
    var = e.get_var_by_attr(dataset_id=dataset, 
                    standard_name=lambda v: str(v).lower() == standard_name)[0]
    download_url = e.get_download_url(dataset_id=dataset, 
                                  variables=['time',var], response='csv', **kwargs)
    df = download_csv(download_url)
    return df, var

map = ipyl.Map(center=center, zoom=zoom, layout=ipyl.Layout(width='650px', height='350px'))
data, datasets = stdname2geojson(e, initial_standard_name, min_time, max_time, nchar)
feature_layer = ipyl.GeoJSON(data=data)
feature_layer.on_click(click_handler)
map.layers = [map.layers[0], feature_layer]

dt_x = bq.DateScale()
sc_y = bq.LinearScale()

initial_dataset = datasets[0]
kwargs = {'time%3E=': min_time, 'time%3C=': max_time}
df, var = get_data(initial_dataset, initial_standard_name, kwargs)
time_series = bq.Lines(x=df.index, y=df[var], scales={'x': dt_x, 'y': sc_y})
ax_x = bq.Axis(scale=dt_x, label='Time')
ax_y = bq.Axis(scale=sc_y, orientation='vertical')
figure = bq.Figure(marks=[time_series], axes=[ax_x, ax_y])
figure.title = '{} - {}'.format(initial_dataset[:nchar], var)
figure.layout.height = '300px'
figure.layout.width = '800px'

ipyw.VBox([dpdown, map, figure])



