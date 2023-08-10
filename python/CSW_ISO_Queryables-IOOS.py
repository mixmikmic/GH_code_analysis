from pylab import *
from owslib.csw import CatalogueServiceWeb
from owslib.sos import SensorObservationService
from owslib import fes
import netCDF4
import pandas as pd
import datetime as dt
from IPython.core.display import HTML

HTML('<iframe src=http://www.nodc.noaa.gov/geoportal/ width=950 height=400></iframe>')

# connect to CSW, explore it's properties

endpoint = 'http://www.ngdc.noaa.gov/geoportal/csw' # NGDC Geoportal

#endpoint = 'http://www.nodc.noaa.gov/geoportal/csw'   # NODC Geoportal: granule level
#endpoint = 'http://data.nodc.noaa.gov/geoportal/csw'  # NODC Geoportal: collection level   
#endpoint = 'http://geodiscover.cgdi.ca/wes/serviceManagerCSW/csw'  # NRCAN CUSTOM
#endpoint = 'http://geoport.whoi.edu/gi-cat/services/cswiso' # USGS Woods Hole GI_CAT
#endpoint = 'http://cida.usgs.gov/gdp/geonetwork/srv/en/csw' # USGS CIDA Geonetwork
#endpoint = 'http://cmgds.marine.usgs.gov/geonetwork/srv/en/csw' # USGS Coastal and Marine Program
#endpoint = 'http://geoport.whoi.edu/geoportal/csw' # USGS Woods Hole Geoportal 
#endpoint = 'http://geo.gov.ckan.org/csw'  # CKAN testing site for new Data.gov
#endpoint = 'https://edg.epa.gov/metadata/csw'  # EPA
#endpoint = 'http://cwic.csiss.gmu.edu/cwicv1/discovery'  # CWIC

csw = CatalogueServiceWeb(endpoint,timeout=60)
csw.version



HTML('<iframe src=https://geo-ide.noaa.gov/wiki/index.php?title=ESRI_Geoportal#PacIOOS_WAF width=950 height=350></iframe>')

regionids = {'AOOS':	'{1E96581F-6B73-45AD-9F9F-2CC3FED76EE6}',
'CENCOOS':	'{BE483F24-52E7-4DDE-909F-EE8D4FF118EA}',
'CARICOOS':	'{0C4CA8A6-5967-4590-BFE0-B8A21CD8BB01}',
'GCOOS':	'{E77E250D-2D65-463C-B201-535775D222C9}',
'GLOS':	'{E4A9E4F4-78A4-4BA0-B653-F548D74F68FA}',
'MARACOOS':	'{A26F8553-798B-4B1C-8755-1031D752F7C2}',
'NANOOS':	'{C6F4754B-30DC-459E-883A-2AC79DA977AB}',
'NAVY':	'{FB160233-7C3B-4841-AD4B-EB5AD843E743}',
'NDBC':	'{B3F50F38-3DE4-4EC9-ABF8-955887829FCC}',
'NERACOOS':	'{E13C88D9-3FF3-4232-A379-84B6A1D7083E}',
'NOS/CO-OPS':	'{2F58127E-A139-4A45-83F2-9695FB704306}',
'PacIOOS':	'{78C0463E-2FCE-4AB2-A9C9-6A34BF261F52}',
'SCCOOS':	'{20A3408F-9EC4-4B36-8E10-BBCDB1E81BDF}',
'SECOORA':	'{E796C954-B248-4118-896C-42E6FAA6EDE9}',
'USACE':	'{4C080A33-F3C3-4F27-AF16-F85BF3095C41}',
'USGS/CMGP': '{275DFB94-E58A-4157-8C31-C72F372E72E}'}

[op.name for op in csw.operations]

def dateRange(start_date='1900-01-01',stop_date='2100-01-01',constraint='overlaps'):
    if constraint == 'overlaps':
        start = fes.PropertyIsLessThanOrEqualTo(propertyname='startDate', literal=stop_date)
        stop = fes.PropertyIsGreaterThanOrEqualTo(propertyname='endDate', literal=start_date)
    elif constraint == 'within':
        start = fes.PropertyIsGreaterThanOrEqualTo(propertyname='startDate', literal=start_date)
        stop = fes.PropertyIsLessThanOrEqualTo(propertyname='endDate', literal=stop_date)
    return start,stop

# get specific ServiceType URL from records
def service_urls(records,service_string='urn:x-esri:specification:ServiceType:odp:url'):
    urls=[]
    for key,rec in records.iteritems():
        #create a generator object, and iterate through it until the match is found
        #if not found, gets the default value (here "none")
        url = next((d['url'] for d in rec.references if d['scheme'] == service_string), None)
        if url is not None:
            urls.append(url)
    return urls

# Perform the CSW query, using Kyle's cool new filters on ISO queryables
# find all datasets in a bounding box and temporal extent that have 
# specific keywords and also can be accessed via OPeNDAP  

box=[-89.0, 30.0, -87.0, 31.0]
start_date='2013-08-21'
stop_date='2013-08-30'
std_name = 'temperature'
service_type='SOS'
region_id = regionids['GCOOS']

# convert User Input into FES filters
start,stop = dateRange(start_date,stop_date,constraint='overlaps')
bbox = fes.BBox(box)
keywords = fes.PropertyIsLike(propertyname='anyText', literal=std_name)
serviceType = fes.PropertyIsLike(propertyname='apiso:ServiceType', literal=('*%s*' % service_type))
siteid = fes.PropertyIsEqualTo(propertyname='sys.siteuuid', literal=region_id)

# try simple query with serviceType and keyword first
csw.getrecords2(constraints=[[serviceType,keywords]],maxrecords=15,esn='full')
for rec,item in csw.records.iteritems():
    print item.title

# try simple query with serviceType and keyword first
csw.getrecords2(constraints=[[serviceType,keywords]],maxrecords=15,esn='full')
for rec,item in csw.records.iteritems():
    print item.title

# check out references for one of the returned records
csw.records['NOAA.NOS.CO-OPS SOS'].references

# filter for GCOOS SOS data
csw.getrecords2(constraints=[[keywords,serviceType,siteid]],maxrecords=15,esn='full')
for rec,item in csw.records.iteritems():
    print item.title

# filter for SOS data in BBOX
csw.getrecords2(constraints=[[keywords,serviceType,bbox]],maxrecords=15,esn='full')
for rec,item in csw.records.iteritems():
    print item.title

urls = service_urls(csw.records,service_string='urn:x-esri:specification:ServiceType:sos:url')
print "\n".join(urls)

urls = [url for url in urls if 'oostethys' not in url]
print "\n".join(urls)

sos = SensorObservationService(urls[0])

getob = sos.get_operation_by_name('getobservation')

print getob.parameters

off = sos.offerings[1]
offerings = [off.name]
responseFormat = off.response_formats[0]
observedProperties = [off.observed_properties[0]]

print sos.offerings[0]



