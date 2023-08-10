from pylab import *
from owslib.csw import CatalogueServiceWeb

from IPython.core.display import HTML
HTML('<iframe src=http://geo.gov.ckan.org/dataset width=1024 height=600></iframe>')

# connect to CSW, explore it's properties
#endpoint = 'http://www.ngdc.noaa.gov/geoportal/csw' # NGDC Geoportal
#endpoint = 'http://www.nodc.noaa.gov/geoportal/csw'   # NODC Geoportal: granule level
#endpoint = 'http://data.nodc.noaa.gov/geoportal/csw'  # NODC Geoportal: collection level
    
#endpoint = 'http://geodiscover.cgdi.ca/wes/serviceManagerCSW/csw'  # NRCAN CUSTOM
#endpoint = 'http://geoport.whoi.edu/gi-cat/services/cswiso' # USGS Woods Hole GI_CAT
#endpoint = 'http://cida.usgs.gov/gdp/geonetwork/srv/en/csw' # USGS CIDA Geonetwork

#endpoint = 'http://geoport.whoi.edu/geoportal/csw'
endpoint = 'http://geo.gov.ckan.org/csw'

csw = CatalogueServiceWeb(endpoint,timeout=30)
csw.version

[op.name for op in csw.operations]

# Perform the CSW query.  To get the Data URLS (including DAP), we need to specify
# esn='full' to get the full Dublin Core response ('summary' is the default)

bbox=[-71.5, 39.5, -63.0, 46]
#bbox=[-180, 0, 180.0, 90]
keywords=['temperature','netcdf']
maxrecords = 10

csw.getrecords(keywords=keywords,bbox=bbox,maxrecords=maxrecords,esn='full')
csw.results

for rec,item in csw.records.iteritems():
    print item.title

for rec,item in csw.records.iteritems():
    print item.references

# function to get specific ServiceType URL from records
def service_urls(records,service_string='urn:x-esri:specification:ServiceType:odp:url'):
    urls=[]
    for key,rec in records.iteritems():
        #create a generator object, and iterate through it until the match is found
        #if not found, gets the default value (here "none")
        url = next((d['url'] for d in rec.references if d['scheme'] == service_string), None)
        if url is not None:
            urls.append(url)
    return urls

urls = service_urls(csw.records,service_string='None')
print(urls)
len(urls)

# examine the 1st record
a=csw.records.keys()
foo=csw.records[a[0]]

foo.abstract

foo.references

foo.xml



