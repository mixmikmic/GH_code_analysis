from owslib.csw import CatalogueServiceWeb
from owslib import fes
import numpy as np

#endpoint = 'http://data.nodc.noaa.gov/geoportal/csw'
endpoint = 'http://www.nodc.noaa.gov/geoportal/csw' 

val = 'wms'
filter2 = fes.PropertyIsLike(propertyname='apiso:ServiceType',literal=('*%s*' % val),
                        escapeChar='\\',wildCard='*',singleChar='?')
csw = CatalogueServiceWeb(endpoint,timeout=60)


filter_list = [filter2]
csw.getrecords2(constraints=filter_list, maxrecords=1000)

print(len(csw.records.keys()))
choice=np.random.choice(list(csw.records.keys()))
print(csw.records[choice].title)
csw.records[choice].references

csw.request

endpoint = 'http://geoport.whoi.edu/csw'   # NODC/UAF Geoportal: granule level

val = 'WMS'
filter2 = fes.PropertyIsLike(propertyname='apiso:ServiceType',literal=('*%s*' % val),
                        escapeChar='\\',wildCard='*',singleChar='?')
csw = CatalogueServiceWeb(endpoint,timeout=60)


filter_list = [filter2]
csw.getrecords2(constraints=filter_list, maxrecords=1000)

print(len(csw.records.keys()))
choice=np.random.choice(list(csw.records.keys()))
print(csw.records[choice].title)
csw.records[choice].references

csw.request

endpoint = 'http://www.ngdc.noaa.gov/geoportal/csw' 

val = 'WMS'
filter2 = fes.PropertyIsLike(propertyname='apiso:ServiceType',literal=('*%s*' % val),
                        escapeChar='\\',wildCard='*',singleChar='?')
csw = CatalogueServiceWeb(endpoint,timeout=60)


filter_list = [filter2]
csw.getrecords2(constraints=filter_list, maxrecords=1000)

print(len(csw.records.keys()))
choice=np.random.choice(list(csw.records.keys()))
print(csw.records[choice].title)
print(csw.records[choice].references)

csw.request





