from owslib.csw import CatalogueServiceWeb
from owslib import fes
import numpy as np

endpoint = 'http://geoport.whoi.edu/csw'
#endpoint = 'http://catalog.data.gov/csw-all'
#endpoint = 'http://www.ngdc.noaa.gov/geoportal/csw'
#endpoint = 'http://www.nodc.noaa.gov/geoportal/csw'
csw = CatalogueServiceWeb(endpoint,timeout=60)
print csw.version

csw.get_operation_by_name('GetRecords').constraints

try:
    csw.get_operation_by_name('GetDomain')
    csw.getdomain('apiso:ServiceType', 'property')
    print(csw.results['values'])
except:
    print('GetDomain not supported')

val = 'COAWST'
filter1 = fes.PropertyIsLike(propertyname='apiso:AnyText',literal=('*%s*' % val),
                        escapeChar='\\',wildCard='*',singleChar='?')
filter_list = [ filter1 ]

csw.getrecords2(constraints=filter_list,maxrecords=100,esn='full')
print len(csw.records.keys())
for rec in list(csw.records.keys()):
    print csw.records[rec].title 
    

choice=np.random.choice(list(csw.records.keys()))
print(csw.records[choice].title)
csw.records[choice].references

val = 'COAWST'
filter1 = fes.PropertyIsLike(propertyname='apiso:AnyText',literal=('*%s*' % val),
                        escapeChar='\\',wildCard='*',singleChar='?',matchCase=True)
val = 'WMS'
filter2 = fes.PropertyIsLike(propertyname='apiso:ServiceType',literal=('*%s*' % val),
                        escapeChar='\\',wildCard='*',singleChar='?',matchCase=False)
filter_list = [ [filter1, filter2] ]
csw.getrecords2(constraints=filter_list, maxrecords=1000)
print(len(csw.records.keys()))

for rec in list(csw.records.keys()):
    print('title:'+csw.records[rec].title) 
    print('identifier:'+csw.records[rec].identifier)
    print('modified:'+csw.records[rec].modified)

val = 'wms'
filter2 = fes.PropertyIsLike(propertyname='apiso:ServiceType',literal=('*%s*' % val),
                        escapeChar='\\',wildCard='*',singleChar='?',matchCase=True)
filter_list = [  filter2 ]
csw.getrecords2(constraints=filter_list, maxrecords=1000)
print(len(csw.records.keys()))



