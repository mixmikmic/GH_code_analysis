from owslib.csw import CatalogueServiceWeb
from owslib import fes
import numpy as np

endpoint = 'http://cinergi.cloudapp.net/geoportal/csw'   # NODC/UAF Geoportal: granule level
csw = CatalogueServiceWeb(endpoint,timeout=60)
print csw.version

csw.get_operation_by_name('GetRecords').constraints

val = 'roms'
#val = 'modflow'
filter1 = fes.PropertyIsLike(propertyname='apiso:AnyText',literal=('*%s*' % val),
                        escapeChar='\\',wildCard='*',singleChar='?')
filter_list = [ filter1 ]

csw.getrecords2(constraints=filter_list,maxrecords=100,esn='full')
print len(csw.records.keys())
for rec in list(csw.records.keys()):
    print csw.records[rec].title 
    

#bbox = [-87.40, 34.25, -63.70, 66.70]    # [lon_min, lat_min, lon_max, lat_max]      maine
bbox = [-162., 19.0, -154., 23.0]   # Hawaii
bbox_filter = fes.BBox(bbox,crs='urn:ogc:def:crs:OGC:1.3:CRS84')
filter_list = [fes.And([filter1, bbox_filter])]
csw.getrecords2(constraints=filter_list, maxrecords=1000)

print(len(csw.records.keys()))
for rec in list(csw.records.keys()):
    print('title:'+csw.records[rec].title) 
    print('identifier:'+csw.records[rec].identifier)
    print('modified:'+csw.records[rec].modified)
    print(' ')

val = 'opendap'
filter2 = fes.PropertyIsLike(propertyname='apiso:ServiceType',literal=('*%s*' % val),
                        escapeChar='\\',wildCard='*',singleChar='?')
filter_list = [fes.And([filter1, filter2, bbox_filter])]
csw.getrecords2(constraints=filter_list, maxrecords=1000, esn='full')

print(len(csw.records.keys()))
for rec in list(csw.records.keys()):
    print('title:'+csw.records[rec].title) 
    print('identifier:'+csw.records[rec].identifier)
    print('modified:'+csw.records[rec].modified)
    print(' ')

choice=np.random.choice(list(csw.records.keys()))
print(csw.records[choice].title)
csw.records[choice].references





