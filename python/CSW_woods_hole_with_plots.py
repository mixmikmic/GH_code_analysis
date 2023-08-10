from pylab import *
from owslib.csw import CatalogueServiceWeb
import netCDF4
import pandas as pd

from IPython.core.display import HTML
HTML('<iframe src=http://geoport.whoi.edu/geoportal/ width=900 height=400></iframe>')

# connect to CSW, explore it's properties
#endpoint = 'http://www.ngdc.noaa.gov/geoportal/csw' # NGDC Geoportal
#endpoint = 'http://www.nodc.noaa.gov/geoportal/csw'   # NODC Geoportal: granule level
#endpoint = 'http://data.nodc.noaa.gov/geoportal/csw'  # NODC Geoportal: collection level
    
#endpoint = 'http://geodiscover.cgdi.ca/wes/serviceManagerCSW/csw'  # NRCAN CUSTOM
#endpoint = 'http://geoport.whoi.edu/gi-cat/services/cswiso' # USGS Woods Hole GI_CAT
#endpoint = 'http://cida.usgs.gov/gdp/geonetwork/srv/en/csw' # USGS CIDA Geonetwork

endpoint = 'http://geoport.whoi.edu/geoportal/csw'

csw = CatalogueServiceWeb(endpoint,timeout=30)
csw.version

[op.name for op in csw.operations]

# Perform the CSW query.  To get the Data URLS (including DAP), we need to specify
# esn='full' to get the full Dublin Core response ('summary' is the default)

#bbox=[-71.5, 39.5, -63.0, 46]
bbox=[-180, 0, 180.0, 90]
std_name='sea_water_temperature'
csw.getrecords(keywords=[std_name],bbox=bbox,maxrecords=5,esn='full')
csw.results

csw.records.keys()

for rec,item in csw.records.iteritems():
    print item.title

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

dap_urls = service_urls(csw.records,service_string='urn:x-esri:specification:ServiceType:odp:url')
print(dap_urls)

def standard_names(nc):
    '''
    get dictionary of variables with standard_names
    '''
    d={}
    for k,v in nc.iteritems():
        try:
            standard_name=v.getncattr('standard_name')
            try:
                d[standard_name]=[d[standard_name],[k]]
            except:
                d[standard_name]=[k]
        except:
            pass
    return d

for url in dap_urls:
    nc = netCDF4.Dataset(url).variables
    lat = nc['lat'][:]
    lon = nc['lon'][:]
    time_var = nc['time']
    dtime = netCDF4.num2date(time_var[:],time_var.units)
    # make a dictionary containing all data from variables that matched the standard_name
    # find list of variables for each standard_name
    d = standard_names(nc)
    # find all the variables matching standard_name=std_name
    d[std_name]
    # read all the data into a dictionary
    data_dict={}
    for v in d[std_name]:
        data_dict[v]=nc[v][:].flatten()
    # Create Pandas data frame, with time index
    ts = pd.DataFrame.from_dict(data_dict)
    ts.index=dtime
    ts.plot(figsize=(12,4));
    title(std_name)

