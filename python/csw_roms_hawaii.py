import os
import sys
from datetime import datetime, timedelta
import dateutil.parser

ioos_tools = os.path.join(os.path.pardir)
sys.path.append(ioos_tools)

#Bounding Box
min_lon, min_lat = -166.0, 19.0 
max_lon, max_lat = -157.0, 23.0 

bbox = [min_lon, min_lat, max_lon, max_lat]
crs = 'urn:ogc:def:crs:OGC:1.3:CRS84'

# Temporal range: Last week.
now = datetime.utcnow()
start, stop = now - timedelta(days=(7)), now

# Temporal range:  Specified time range
start = dateutil.parser.parse('2017-03-01T00:00:00Z')
stop  = dateutil.parser.parse('2017-04-01T00:00:00Z')

# Find any of these Ocean Model Names
model_names = ['ROMS', 'FVCOM', 'SELFE', 'ADCIRC', 'Delft3D', 'DelftFM', 'HyCOM', 'NCOM']

# ServiceType
service_type = 'WMS'

from owslib import fes
from ioos_tools.ioos import fes_date_filter

kw = dict(wildCard='*', escapeChar='\\',
          singleChar='?', propertyname='apiso:AnyText')

or_filt = fes.Or([fes.PropertyIsLike(literal=('*%s*' % val), **kw)
                  for val in model_names])

kw = dict(wildCard='*', escapeChar='\\',
          singleChar='?', propertyname='apiso:ServiceType')

serviceType = fes.PropertyIsLike(literal=('*%s*' % service_type), **kw)


begin, end = fes_date_filter(start, stop)
bbox_crs = fes.BBox(bbox, crs=crs)

filter_list = [
    fes.And(
        [
            bbox_crs,  # bounding box
            begin, end,  # start and end date
            or_filt,  # or conditions (CF variable names)
            serviceType  # search only for datasets that have WMS services
        ]
    )
]

from owslib.csw import CatalogueServiceWeb


endpoint = 'https://data.ioos.us/csw'

csw = CatalogueServiceWeb(endpoint, timeout=60)

def get_csw_records(csw, filter_list, pagesize=10, maxrecords=1000):
    """Iterate `maxrecords`/`pagesize` times until the requested value in
    `maxrecords` is reached.
    """
    from owslib.fes import SortBy, SortProperty
    # Iterate over sorted results.
    sortby = SortBy([SortProperty('dc:title', 'ASC')])
    csw_records = {}
    startposition = 0
    nextrecord = getattr(csw, 'results', 1)
    while nextrecord != 0:
        csw.getrecords2(constraints=filter_list, startposition=startposition,
                        maxrecords=pagesize, sortby=sortby)
        csw_records.update(csw.records)
        if csw.results['nextrecord'] == 0:
            break
        startposition += pagesize + 1  # Last one is included.
        if startposition >= maxrecords:
            break
    csw.records.update(csw_records)

get_csw_records(csw, filter_list, pagesize=10, maxrecords=1000)

records = '\n'.join(csw.records.keys())
print('Found {} records.\n'.format(len(csw.records.keys())))
for key, value in list(csw.records.items()):
    print('[{}]\n{}\n'.format(value.title, key))

csw.request

csw_request = '"{}": {}"'.format('getRecordsTemplate',str(csw.request,'utf-8'))

import io
import json
with io.open('query.json', 'a', encoding='utf-8') as f:
            f.write(json.dumps(csw_request, ensure_ascii=False))
            f.write('\n')

filter_list = [ or_filt ]

get_csw_records(csw, filter_list, pagesize=10, maxrecords=1000)

records = '\n'.join(csw.records.keys())
print('Found {} records.\n'.format(len(csw.records.keys())))
k=0
for key, value in list(csw.records.items()):
    print('[{}]\n{}\n'.format(value.title, key))
    [print(d['scheme']) for d in value.references]
    print('\n')
    if any('None' in d['scheme'] for d in value.references):
        k = k+1

[print(d['scheme']) for d in value.references]

k

import textwrap

print('\n'.join(textwrap.wrap(value.abstract)))

print('\n'.join(value.subjects))

from geolinks import sniff_link

msg = 'geolink: {geolink}\nscheme: {scheme}\nURL: {url}\n'.format
for ref in value.references:
    print(msg(geolink=sniff_link(ref['url']), **ref))

start, stop

for ref in value.references:
    url = ref['url']
    if 'csv' in url and 'sea' in url and 'temperature' in url:
        print(msg(geolink=sniff_link(url), **ref))
        break

fmt = ('http://opendap.co-ops.nos.noaa.gov/ioos-dif-sos/SOS?'
       'service=SOS&'
       'eventTime={0:%Y-%m-%dT00:00:00}/{1:%Y-%m-%dT00:00:00}&'
       'observedProperty=http://mmisw.org/ont/cf/parameter/sea_water_temperature&'
       'version=1.0.0&'
       'request=GetObservation&offering=urn:ioos:station:NOAA.NOS.CO-OPS:9439040&'
       'responseFormat=text/csv')

url = fmt.format(start, stop)

import io
import requests
import pandas as pd

r = requests.get(url)

df = pd.read_csv(io.StringIO(r.content.decode('utf-8')),
                 index_col='date_time', parse_dates=True)

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt


fig, ax = plt.subplots(figsize=(11, 2.75))
ax = df['sea_water_temperature (C)'].plot(ax=ax)
ax.set_xlabel('')
ax.set_ylabel(r'Sea water temperature ($^\circ$C)')
ax.set_title(value.title)

