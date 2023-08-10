import os
import sys
import time
import warnings

ioos_tools = os.path.join(os.path.pardir)
sys.path.append(ioos_tools)

from datetime import datetime, timedelta

# Region.
bbox = [-123, 36, -121, 40]
crs = 'urn:ogc:def:crs:OGC:1.3:CRS84'
    
# Temporal range.
now = datetime.utcnow()
start,  stop = now - timedelta(days=(7)), now

# Names.
cf_names = ['*sea_water_potential_temperature*',
            '*sea_water_salinity*']

from owslib import fes
from ioos_tools.ioos import fes_date_filter

kw = dict(wildCard='*', escapeChar='\\',
          singleChar='?', propertyname='apiso:AnyText')

or_filt = fes.Or([fes.PropertyIsLike(literal=('*%s*' % val), **kw)
                  for val in cf_names])

# Exclude ROMS Averages and History files.
not_filt = fes.Not([fes.PropertyIsLike(literal='*GNOME*', **kw)])

services = ['OPeNDAP','SOS'] 
#services = ['asdfasfasdf','asfasdfasdfasd']
service_filt = fes.Or([fes.PropertyIsLike(propertyname='apiso:ServiceType',literal=('*%s*' % val),
                        escapeChar='\\',wildCard='*',singleChar='?') for val in services])

begin, end = fes_date_filter(start, stop)
bbox_crs = fes.BBox(bbox, crs=crs)
filter_list = [fes.And([bbox_crs, begin, end, or_filt, not_filt])]
filter_list = [fes.And([or_filt, not_filt])]
filter_list = [fes.And([or_filt, not_filt, service_filt])]

from owslib.csw import CatalogueServiceWeb


catalogs = ['http://www.ngdc.noaa.gov/geoportal/csw',
            'https://dev-catalog.ioos.us/csw',
            'http://geoport.whoi.edu/csw',
            'http://catalog.data.gov/csw-all']

catalogs = ['https://dev-catalog.ioos.us/csw',
            'http://geoport.whoi.edu/csw',
            'http://catalog.data.gov/csw-all']

for endpoint in catalogs:
    csw = CatalogueServiceWeb(endpoint, timeout=60)
    csw.getrecords2(constraints=filter_list, maxrecords=1000, esn='full')
    print(endpoint)
    print(len(csw.records))
    print(csw.records.keys())





