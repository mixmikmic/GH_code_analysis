"""
The original notebook is HF_Radar_Currents_west_coast.ipynb in
Theme_2_Extreme_Events/Scenario_2B/HF_Radar_Currents/
"""

title = "Near real-time current data"
name = '2015-10-26-HF_Radar_Currents_west_coast'

get_ipython().magic('matplotlib inline')
import seaborn
seaborn.set(style='ticks')

import os
from datetime import datetime
from IPython.core.display import HTML

import warnings
warnings.simplefilter("ignore")

# Metadata and markdown generation.
hour = datetime.utcnow().strftime('%H:%M')
comments = "true"

date = '-'.join(name.split('-')[:3])
slug = '-'.join(name.split('-')[3:])

metadata = dict(title=title,
                date=date,
                hour=hour,
                comments=comments,
                slug=slug,
                name=name)

markdown = """Title: {title}
date:  {date} {hour}
comments: {comments}
slug: {slug}

{{% notebook {name}.ipynb cells[2:] %}}
""".format(**metadata)

content = os.path.abspath(os.path.join(os.getcwd(), os.pardir,
                                       os.pardir, '{}.md'.format(name)))

with open('{}'.format(content), 'w') as f:
    f.writelines(markdown)


html = """
<small>
<p> This post was written as an IPython notebook.  It is available for
<a href="http://ioos.github.com/system-test/downloads/
notebooks/%s.ipynb">download</a>.  You can also try an interactive version on
<a href="http://mybinder.org/repo/ioos/system-test/">binder</a>.</p>
<p></p>
""" % (name)

from datetime import datetime, timedelta


bounding_box_type = "box"
bounding_box = [-123, 36, -121, 40]

# Temporal range.
now = datetime.utcnow()
start,  stop = now - timedelta(days=(7)), now

# URL construction.
iso_start = start.strftime('%Y-%m-%dT%H:%M:00Z')
iso_end = stop.strftime('%Y-%m-%dT%H:%M:00Z')

print('{} to {}'.format(iso_start, iso_end))

data_dict = dict()
sos_name = 'Currents'
data_dict['currents'] = {"names": ['currents',
                                   'surface_eastward_sea_water_velocity',
                                   '*surface_eastward_sea_water_velocity*'],
                         "sos_name": [sos_name]}

from owslib.csw import CatalogueServiceWeb


endpoint = 'http://www.ngdc.noaa.gov/geoportal/csw'

csw = CatalogueServiceWeb(endpoint, timeout=60)

from owslib import fes
from utilities import fes_date_filter
                       
begin, end = fes_date_filter(start, stop)
bbox = fes.BBox(bounding_box)

or_filt = fes.Or([fes.PropertyIsLike(propertyname='apiso:AnyText',
                                     literal=('*%s*' % val),
                                     escapeChar='\\', wildCard='*',
                                     singleChar='?') for val in
                  data_dict['currents']['names']])

val = 'Averages'
not_filt = fes.Not([fes.PropertyIsLike(propertyname='apiso:AnyText',
                                       literal=('*%s*' % val),
                                       escapeChar='\\',
                                       wildCard='*',
                                       singleChar='?')])

filter_list = [fes.And([bbox, begin, end, or_filt, not_filt])]

csw.getrecords2(constraints=filter_list, maxrecords=1000, esn='full')

from utilities import service_urls

dap_urls = service_urls(csw.records, service='odp:url')

print("Total DAP: {}".format(len(dap_urls)))
print("\n".join(dap_urls))

from pandas import read_csv

box_str = ','.join(str(e) for e in bounding_box)

url = (('http://opendap.co-ops.nos.noaa.gov/ioos-dif-sos/SOS?'
        'service=SOS&request=GetObservation&version=1.0.0&'
        'observedProperty=%s&bin=1&'
        'offering=urn:ioos:network:NOAA.NOS.CO-OPS:CurrentsActive&'
        'featureOfInterest=BBOX:%s&responseFormat=text/csv') % (sos_name,
                                                                box_str))

obs_loc_df = read_csv(url)

from utilities.ioos import processStationInfo

st_list = processStationInfo(obs_loc_df, "coops")

print(st_list)

box_str = ','.join(str(e) for e in bounding_box)

url = (('http://sdf.ndbc.noaa.gov/sos/server.php?'
        'request=GetObservation&service=SOS&'
        'version=1.0.0&'
        'offering=urn:ioos:network:noaa.nws.ndbc:all&'
        'featureofinterest=BBOX:%s&'
        'observedproperty=%s&'
        'responseformat=text/csv&') % (box_str, sos_name))

obs_loc_df = read_csv(url)

len(obs_loc_df)

import uuid
from IPython.display import HTML, Javascript, display
from utilities.ioos import coopsCurrentRequest, ndbcSOSRequest


divid = str(uuid.uuid4())
pb = HTML("""
<div style="border: 1px solid black; width:500px">
  <div id="%s" style="background-color:blue; width:0%%">&nbsp;</div>
</div>
""" % divid)

display(pb)

count = 0
for station_index in st_list.keys():
    st = station_index.split(":")[-1]
    tides_dt_start = start.strftime('%Y%m%d %H:%M')
    tides_dt_end = stop.strftime('%Y%m%d %H:%M')

    if st_list[station_index]['source'] == "coops":
        df = coopsCurrentRequest(st, tides_dt_start, tides_dt_end)
    elif st_list[station_index]['source'] == "ndbc":
        df = ndbcSOSRequest(station_index, date_range_string)

    if (df is not None) and (len(df) > 0):
        st_list[station_index]['hasObsData'] = True
    else:
        st_list[station_index]['hasObsData'] = False
    st_list[station_index]['obsData'] = df

    print('{}'.format(station_index))

    count += 1
    percent_compelte = (float(count)/float(len(st_list.keys()))) * 100
    display(Javascript("$('div#%s').width('%i%%')" %
                       (divid, int(percent_compelte))))

from utilities.ioos import get_hr_radar_dap_data

df_list = get_hr_radar_dap_data(dap_urls, st_list, start, stop)

def findSFOIndexs(lats, lons, lat_lon_list):
    index_list, dist_list = [], []
    for val in lat_lon_list:
        point1 = Point(val[1], val[0])
        dist = 999999999
        index = -1
        for k in range(0, len(lats)):
            point2 = Point(lons[k], lats[k])
            val = point1.distance(point2)
            if val < dist:
                index = k
                dist = val
        index_list.append(index)
        dist_list.append(dist)
    print(index_list, dist_list)
    return index_list, dist_list


def buildSFOUrls(start,  stop):
    """
    Multiple files for time step, were only looking at Nowcast (past) values
    times are 3z, 9z, 15z, 21z

    """
    url_list = []
    time_list = ['03z', '09z', '15z', '21z']
    delta = stop - start
    for i in range((delta.days)+1):
        model_file_date = start + timedelta(days=i)
        base_url = ('http://opendap.co-ops.nos.noaa.gov/'
                    'thredds/dodsC/NOAA/SFBOFS/MODELS/')
        val_month, val_year, val_day = '', '', ''
        # Month.
        if model_file_date.month < 10:
            val_month = "0" + str(model_file_date.month)
        else:
            val_month = str(model_file_date.month)
        # Year.
        val_year = str(model_file_date.year)
        # Day.
        if model_file_date.day < 10:
            val_day = "0" + str(model_file_date.day)
        else:
            val_day = str(model_file_date.day)
        file_name = '/nos.sfbofs.stations.nowcast.'
        file_name += val_year + val_month + val_day
        for t in time_list:
            t_val = '.t' + t + '.nc'
            url_list.append(base_url + val_year + val_month +
                            file_name + t_val)
    return url_list

from utilities.ioos import extractSFOModelData

st_known = st_list.keys()

name_list, lat_lon_list = [], []
for st in st_list:
    if st in st_known:
        lat = st_list[st]['lat']
        lon = st_list[st]['lon']
        name_list.append(st)
        lat_lon_list.append([lat, lon])

model_data = extractSFOModelData(lat_lon_list, name_list, start, stop)

import matplotlib.pyplot as plt


for station_index in st_list.keys():
    df = st_list[station_index]['obsData']
    if st_list[station_index]['hasObsData']:
        fig = plt.figure(figsize=(16, 3))
        plt.plot(df.index, df['sea_water_speed (cm/s)'])
        fig.suptitle('Station:'+station_index, fontsize=14)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('sea_water_speed (cm/s)', fontsize=14)

        if station_index in model_data:
            df_model = model_data[station_index]['data']
            plt.plot(df_model.index, df_model['sea_water_speed (cm/s)'])
        for ent in df_list:
            if ent['ws_pts'] > 4:
                if station_index == ent['name']:
                    df = ent['data']
                    plt.plot(df.index, df['sea_water_speed (cm/s)'])
                    ent['valid'] = True
    l = plt.legend(('Station Obs', 'Model', 'HF Radar'), loc='upper left')

import folium
from utilities import get_coordinates

station = st_list[list(st_list.keys())[0]]
mapa = folium.Map(location=[station["lat"], station["lon"]], zoom_start=10)
mapa.line(get_coordinates(bounding_box),
          line_color='#FF0000',
          line_weight=5)

for st in st_list:
    lat = st_list[st]['lat']
    lon = st_list[st]['lon']

    popupString = ('<b>Obs Location:</b><br>' + st + '<br><b>Source:</b><br>' +
                   st_list[st]['source'])

    if 'hasObsData' in st_list[st] and st_list[st]['hasObsData'] == False:
        mapa.circle_marker([lat, lon], popup=popupString, radius=1000,
                           line_color='#FF0000', fill_color='#FF0000',
                           fill_opacity=0.2)

    elif st_list[st]['source'] == "coops":
        mapa.simple_marker([lat, lon], popup=popupString,
                           marker_color="darkblue", marker_icon="star")
    elif st_list[st]['source'] == "ndbc":
        mapa.simple_marker([lat, lon], popup=popupString,
                           marker_color="darkblue", marker_icon="star")

try:
    for ent in df_list:
        lat = ent['lat']
        lon = ent['lon']
        popupstring = ("HF Radar: [" + str(lat) + ":"+str(lon) + "]" +
                       "<br>for<br>" + ent['name'])
        mapa.circle_marker([lat, lon], popup=popupstring, radius=500,
                           line_color='#FF0000', fill_color='#FF0000',
                           fill_opacity=0.5)
except:
    pass

try:
    for st in model_data:
        lat = model_data[st]['lat']
        lon = model_data[st]['lon']
        popupstring = ("HF Radar: [" + str(lat) + ":" + str(lon) + "]" +
                       "<br>for<br>" + ent['name'])
        mapa.circle_marker([lat, lon], popup=popupstring, radius=500,
                           line_color='#66FF33', fill_color='#66FF33',
                           fill_opacity=0.5)
except:
    pass

now = datetime.utcnow()
mapa.add_tile_layer(tile_name='hfradar 2km',
                    tile_url=('http://hfradar.ndbc.noaa.gov/tilesavg.php?'
                              's=10&e=100&x={x}&y={y}&z={z}&t=' +
                              str(now.year) + '-' + str(now.month) +
                              '-' + str(now.day) + ' ' +
                              str(now.hour-2) + ':00:00&rez=2'),
                   attr='NDBC/NOAA')

mapa.add_layers_to_map()

mapa

HTML(html)

