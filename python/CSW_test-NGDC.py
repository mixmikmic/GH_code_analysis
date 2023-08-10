from pylab import *
from owslib.csw import CatalogueServiceWeb
from owslib import fes
import random
import netCDF4
import pandas as pd
import datetime as dt

#from IPython.core.display import HTML
#HTML('<iframe src=http://www.ngdc.noaa.gov/geoportal/ width=950 height=400></iframe>')

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

[op.name for op in csw.operations]

# hopefully something like this will be implemented in fes soon
def dateRange(start_date='1900-01-01',stop_date='2100-01-01',constraint='overlaps'):
    if constraint == 'overlaps':
        start = fes.PropertyIsLessThanOrEqualTo(propertyname='apiso:TempExtent_begin', literal=stop_date)
        stop = fes.PropertyIsGreaterThanOrEqualTo(propertyname='apiso:TempExtent_end', literal=start_date)
    elif constraint == 'within':
        start = fes.PropertyIsGreaterThanOrEqualTo(propertyname='apiso:TempExtent_begin', literal=start_date)
        stop = fes.PropertyIsLessThanOrEqualTo(propertyname='apiso:TempExtent_end', literal=stop_date)
    return start,stop

# hopefully something like this will be implemented in fes soon
def zRange(min_z='-5000', max_z='0', constraint='overlaps'):
    if constraint == 'overlaps':
        zmin = fes.PropertyIsLessThanOrEqualTo(propertyname='apiso:VertExtent_min', literal=min_z)
        zmax = fes.PropertyIsGreaterThanOrEqualTo(propertyname='apiso:VertExtent_max', literal=max_z)
    elif constraint == 'within':
        zmin = fes.PropertyIsGreaterThanOrEqualTo(propertyname='apiso:VertExtent_min', literal=min_z)
        zmax = fes.PropertyIsLessThanOrEqualTo(propertyname='apiso:VertExtent_max', literal=max_z)
    return zmin,zmax

# User Input for query

#box=[-120, 2.0, -110.0, 6.0] #  oceansites
box=[-160, 19, -156, 23]   # pacioos
start_date='2012-05-01'
stop_date='2012-06-01'
min_z = '-5000'
max_z = '0'
responsible_party = 'Jim Potemra'
responsible_party = 'Margaret McManus'
std_name = 'sea_water_temperature'
service_type='opendap'

# convert User Input into FES filters
start,stop = dateRange(start_date,stop_date)
zmin,zmax = zRange(min_z,max_z,constraint='within')
bbox = fes.BBox(box)
#keywords = fes.PropertyIsLike(propertyname='anyText', literal=std_name)
#keywords = fes.PropertyIsLike(propertyname='apiso:Keywords', literal=std_name)
keywords = fes.PropertyIsEqualTo(propertyname='apiso:Keywords', literal=std_name)
serviceType = fes.PropertyIsLike(propertyname='apiso:ServiceType', literal=('*%s*' % service_type))
ResponsiblePartyName = fes.PropertyIsEqualTo(propertyname='apiso:ResponsiblePartyName', literal=responsible_party)
#serviceType = fes.PropertyIsEqualTo(propertyname='apiso:ServiceType', literal=service_type)


# try simple request using only keywords

csw.getrecords2(constraints=[keywords],maxrecords=5)
csw.records.keys()

# try request using multiple filters "and" syntax: [[filter1,filter2]]
csw.getrecords2(constraints=[[keywords,start,stop,serviceType,bbox]],maxrecords=5,esn='full')
csw.records.keys()

# try request using multiple filters "and" syntax: [[filter1,filter2]]
csw.getrecords2(constraints=[[keywords,start,stop,bbox]],maxrecords=5,esn='full')
csw.records.keys()

choice=random.choice(list(csw.records.keys()))
print choice
csw.records[choice].references

# try adding responsible party role 
csw.getrecords2(constraints=[ResponsiblePartyName],maxrecords=10,esn='full')
csw.records.keys()

for rec,item in csw.records.iteritems():
    print item.title

csw.getrecords2(constraints=[[keywords,start,stop,serviceType,bbox,zmin,zmax]],maxrecords=5,esn='full')
csw.records.keys()

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
print "\n".join(dap_urls)

def standard_names(ncv):
    '''
    get dictionary of variables with standard_names
    '''
    d={}
    for k,v in ncv.iteritems():
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
    ncv = netCDF4.Dataset(url).variables
    print ncv.keys()
    lat = ncv['lon'][:]
    lon = ncv['lat'][:]
    tvar = ncv['time']
    istart = netCDF4.date2index(dt.datetime.strptime(start_date,'%Y-%m-%d'),tvar,select='nearest')
    istop = netCDF4.date2index(dt.datetime.strptime(stop_date,'%Y-%m-%d'),tvar,select='nearest')
    if istart != istop:
        dtime = netCDF4.num2date(tvar[istart:istop],tvar.units)
        # make a dictionary containing all data from variables that matched the standard_name
        # find list of variables for each standard_name
        d = standard_names(ncv)
        # find all the variables matching standard_name=std_name
        print d[std_name]
        # read all the data into a dictionary
        data_dict={}
        lev=0
        for v in d[std_name]:
            data_dict[v]=ncv[v][istart:istop].flatten()
        # Create Pandas data frame, with time index
        ts = pd.DataFrame.from_dict(data_dict)
        ts.index=dtime
        ts.plot(figsize=(12,4));
        title(std_name)

