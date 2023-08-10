#Query Geoportal CSW to find COAWST data

from owslib.csw import CatalogueServiceWeb
from owslib import fes
import numpy as np

endpoint = 'http://www.ngdc.noaa.gov/geoportal/csw'  
csw = CatalogueServiceWeb(endpoint,timeout=60)
print csw.version

csw.get_operation_by_name('GetRecords').constraints

val = 'COAWST'
filter1 = fes.PropertyIsLike(propertyname='apiso:AnyText',literal=('*%s*' % val),
                        escapeChar='\\',wildCard='*',singleChar='?')
filter_list = [ filter1 ]

csw.getrecords2(constraints=filter_list,maxrecords=100,esn='full')
print len(csw.records.keys())
for rec in list(csw.records.keys()):
    print('title:'+csw.records[rec].title) 
    print('identifier:'+csw.records[rec].identifier)
    print('modified:'+csw.records[rec].modified)

choice=np.random.choice(list(csw.records.keys()))
print(csw.records[choice].title)
csw.records[choice].references

val = 'wms'
filter1 = fes.PropertyIsLike(propertyname='apiso:ServiceType',literal=('*%s*' % val),
                        escapeChar='\\',wildCard='*',singleChar='?')
filter_list = [ filter1]
csw.getrecords2(constraints=filter_list, maxrecords=1000)
print(len(csw.records.keys()))

val = 'wms'
filter1 = fes.PropertyIsLike(propertyname='apiso:ServiceType',literal=('*%s*' % val),
                        escapeChar='\\',wildCard='*',singleChar='?')
val = 'COAWST'
filter2 = fes.PropertyIsLike(propertyname='apiso:AnyText',literal=('*%s*' % val),
                        escapeChar='\\',wildCard='*',singleChar='?')
filter_list = [ [filter1, filter2] ]
csw.getrecords2(constraints=filter_list, maxrecords=1000)
print(len(csw.records.keys()))

for rec in list(csw.records.keys()):
    print('title:'+csw.records[rec].title) 
    print('identifier:'+csw.records[rec].identifier)
    print('modified:'+csw.records[rec].modified)

val = 'Warner'
#val = 'COADS'
filter1 = fes.PropertyIsLike(propertyname='apiso:anyText',literal=('*%s*' % val),
                        escapeChar='\\',wildCard='*',singleChar='?')
filter_list = [ filter1 ]
csw.getrecords2(constraints=filter_list, maxrecords=1000)



print(len(csw.records.keys()))
for rec in list(csw.records.keys()):
    print('title:'+csw.records[rec].title) 
    print('identifier:'+csw.records[rec].identifier)
    print('modified:'+csw.records[rec].modified)
    print(' ')



