get_ipython().magic('matplotlib inline')

#python standard library imports
import urllib
import sys
import json
from collections import OrderedDict
from datetime import datetime,timedelta
import os.path

#third party library imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap

#constants
BASEURL = 'http://earthquake.usgs.gov/fdsnws/event/1/query?'

indict = {'format':'geojson',
         'starttime':datetime(2011,11,6,3,53,0),
         'endtime':datetime(2011,11,6,3,54,0),
         'minmag':5.5,
         'maxmag':5.7}
#the urllib module (in Python 3) encapsulates everything the standard library knows about urls
params = urllib.parse.urlencode(indict)
#assemble complete url
url = '%s%s' % (BASEURL,params)

#here we're using the request submodule to open a url, just like we would use open() to open a file.
f = urllib.request.urlopen(url)
#urlopen() returns a file-like object, which means that it behaves just like a file object does, including
#allowing you to read in all the data from the thing that has been opened.
#note the decode() method, which is a new necessity in Python 3, in order to convert a string of bytes
#into ASCII (utf-8).
data = f.read().decode('utf-8')
#Always close your file!
f.close()

print(data)

jsondict = json.loads(data)
jsondict

print(jsondict.keys())
print()
for key,value in jsondict['metadata'].items():
    print('%s = %s' % (key,str(value)))
    
print()
print(jsondict['type'])

print()
print(type(jsondict['features']))
print(len(jsondict['features']))

jsondict['features'][0]

durl = jsondict['features'][0]['properties']['detail']
fh = urllib.request.urlopen(durl)
data = fh.read().decode('utf-8')
fh.close()
ejsondict = json.loads(data)
ejsondict

cdi_url = ejsondict['properties']['products']['dyfi'][0]['contents']['cdi_geo.txt']['url']
columns = ['Box','CDI','NumResponses','HypoDistance',
            'Latitude','Longitude','Suspect?','City','State','box2']
table = pd.read_table(cdi_url,sep=',',skiprows=[0],names=columns)
table.head()

def getRequestData(indict):
    """Return the list of earthquake features from a ComCat API search.
    
    This function does no error handling - use at your own risk.
    
    :param indict:
      Dictionary containing any of the parameters described in http://earthquake.usgs.gov/fdsnws/event/1/.
    :returns: 
      List of earthquake features.
    """
    #we are hardcoding format to geojson
    indict['format'] = 'geojson'
    #we want earthquakes to be sorted ascending by time
    indict['orderby'] = 'time-asc'
    copydict = indict.copy()
    for key,value in copydict.items():
        #re-format datetime objects to ISO string representations
        if isinstance(value,datetime):
            copydict[key] = value.strftime('%Y-%m-%dT%H:%M:%S')
            
    #assemble dictionary into url parameters string
    params = urllib.parse.urlencode(indict)
    #assemble complete url
    url = '%s%s' % (BASEURL,params)
    #open a file-like object on the url
    f = urllib.request.urlopen(url)
    #read the data from that file-like object, remembering to decode the bytes into an ASCII JSON string.
    data = f.read().decode('utf-8')
    #parse that JSON string into a data structure (list of dictionaries). 
    features = json.loads(data)['features']
    f.close()
    return features

END = datetime.utcnow()
START = END - timedelta(days=90) #three months back
params = {'starttime':datetime(2016,2,13,17,7,0),
          'endtime':datetime.utcnow(),
          'minmagnitude':4.5,
          'maxmagnitude':9.9,
          'minlatitude':33.49,
          'maxlatitude':37.234,
          'minlongitude':-103.118,
          'maxlongitude':-94.219,
          'producttype':'dyfi',
         }
events = getRequestData(params)
print(len(events))

event = events[0] #events is a list, this gets the first (zeroth) element in the list
print('Event is an instance of %s' % type(event))

print('The keys of the event dictionary are: %s' % list(event.keys()))

print(event['id'])
print(event['type'])
print(event['geometry'])

print('The keys of the event properties dictionary are: \n')
for key,value in event['properties'].items():
    print('%s = %s' % (key,str(value)))

#TODO - write a section exploring the event detail geojson...

def getDYFITable(event):
    """Retrieve a geo-coded DYFI summary file from the event dictionary obtained by getRequestData().
    
    :param event:
      Event dictionary containing fields:
        - id Event ID.
        - geometry Dictionary containing 'coordinates' field with 3-element list of [lon,lat,depth].
        - properties Dictionary containing fields:
          - time Number of milliseconds since Jan 1 1970 (Unix epoch).
          - mag  Magnitude.
          - detail url of event-specific geojson.
    :returns:
      pandas DataFrame containing the contents of the DYFI geocoded Intensity Summary text file, 
      OR
      None, as these files are not compiled for all events with DYFI results.
    """
    #extract all of the event-specific parameters
    eventid = event['id']
    lon,lat,depth = event['geometry']['coordinates']
    time = datetime.utcfromtimestamp(event['properties']['time']//1000)
    mag = event['properties']['mag']
    
    #next open the url for the event-specific page, read it's JSON data, parse into event dictionary. 
    detail = event['properties']['detail']
    fh = urllib.request.urlopen(detail)
    eventdata = fh.read().decode('utf-8')
    fh.close()
    eventjson = json.loads(eventdata)
    
    #make sure dyfi is listed as one of the products
    if 'dyfi' not in eventjson['properties']['products']:
        return None
    
    #make sure the geocoded intensity summary file is included among list of DYFI output.
    dyfistuff = eventjson['properties']['products']['dyfi'][0]['contents']
    if 'cdi_geo.txt' not in dyfistuff:
        return None
    
    #get the url for the intensity summary file
    cdifile = dyfistuff['cdi_geo.txt']['url']
    #read in the text file using the pandas read_table() function, specifying the names of the columns
    columns = ['Box','CDI','NumResponses','HypoDistance',
               'Latitude','Longitude','Suspect?','City','State','box2']
    table = pd.read_table(cdifile,sep=',',skiprows=[0],names=columns)
    
    #box2 appears to be a duplicate of the first column.  Let's drop it.
    table.drop('box2', axis=1, inplace=True)
    
    #Now let's add in the eventid, time, lat,lon,depth,mag columns.
    #These will be duplicated for each row, which is an inefficiency we'll live with
    #for the sake of ease of use.
    nrows = len(table)
    table['EventID'] = [eventid]*nrows
    table['Time'] = [time]*nrows
    table['EventLatitude'] = [lat]*nrows
    table['EventLongitude'] = [lon]*nrows
    table['EventDepth'] = [depth]*nrows
    table['Magnitude'] = [mag]*nrows
    
    #re-order the columns the way we want them
    cols = ['Box','EventID','Time','EventLatitude','EventLongitude','EventDepth',
            'Magnitude','CDI','NumResponses','HypoDistance','Latitude','Longitude',
            'Suspect?','City','State']
    table = table[cols]
    return table

table = getDYFITable(event)
table.head()

params = {'starttime':datetime(2016,1,1),
          'endtime':datetime(2016,7,19),
          'minmagnitude':3.5,
          'maxmagnitude':9.9,
          'minlatitude':33.49,
          'maxlatitude':37.234,
          'minlongitude':-103.118,
          'maxlongitude':-94.219,
          'producttype':'dyfi',
         }
events = getRequestData(params)
print('%i events found between %s and %s' % (len(events),params['starttime'],params['endtime']))

t1 = datetime.now()
df = None
nevents = 0
months = []
for event in events:
    etime = datetime.utcfromtimestamp(event['properties']['time']//1000)
    monstr = etime.strftime('%Y-%m')
    if monstr not in months:
        sys.stdout.write(monstr)
        months.append(monstr)
    table = getDYFITable(event)
    if table is None:
        continue
    nevents += 1
    if df is None:
        sys.stdout.write('.')
        #print('Appending %i rows to DataFrame' % len(table))
        df = table.copy()
    else:
        sys.stdout.write('.')
        #print('Appending %i rows to DataFrame' % len(table))
        df = df.append(table.copy(),ignore_index=True)

print()
t2 = datetime.now()
nsecs = (t2-t1).seconds
print('DataFrame has %i rows from %i events - %i seconds elapsed.' % (len(df),nevents,nsecs))

outfile = os.path.join(os.path.expanduser('~'),'dyfi_oklahoma.csv')
df.to_csv(outfile)
1769/60

df.head()

df2 = df[df.NumResponses >= 3]

len(df2)

umags = np.unique(df2.Magnitude)
print('%i unique magnitudes between %.1f and %.1f' % (len(umags),umags.min(),umags.max()))
print('%s' % (str(umags)))

def floor_to_nearest(value,floor_value=1000):
    """Return the value, floored to nearest floor_value (defaults to 1000).
    
    :param value: 
      Value to be floored.
    :param floor_value: 
      Number to which the value should be floored.
    :returns:
      Floored value.
    """
    if floor_value < 1:
        ds = str(floor_value)
        nd = len(ds) - (ds.find('.')+1)
        value = value * 10**nd
        floor_value = floor_value * 10**nd
        value = int(np.floor(float(value)/floor_value)*floor_value)
        value = float(value) / 10**nd
    else:
        value = int(np.floor(float(value)/floor_value)*floor_value)
    return value

def ceil_to_nearest(value,ceil_value=1000):
    """Return the value, ceiled to nearest ceil_value (defaults to 1000).
    
    :param value: 
      Value to be ceiled.
    :param ceil_value: 
      Number to which the value should be ceiled.
    :returns:
      Ceiled value.
    """
    if ceil_value < 1:
        ds = str(ceil_value)
        nd = len(ds) - (ds.find('.')+1)
        value = value * 10**nd
        ceil_value = ceil_value * 10**nd
        value = int(np.ceil(float(value)/ceil_value)*ceil_value)
        value = float(value) / 10**nd
    else:
        value = int(np.ceil(float(value)/ceil_value)*ceil_value)
    return value

magmin = floor_to_nearest(umags.min(),0.5)
magmax = ceil_to_nearest(umags.max(),0.5)
for maglow in np.arange(magmin,magmax,0.5):
    maghigh = maglow + 0.5
    df_sub = df2[(df2.Magnitude >= maglow) & (df2.Magnitude < maghigh)]
    f = plt.figure();
    plt.semilogx(df_sub.HypoDistance,df_sub.CDI,'b.');
    plt.axis([0,500,0,7])
    plt.title('Distance vs CDI for Magnitude Range %.2f to %.2f (N=%i)' % (maglow,maghigh,len(df_sub)));
    plt.xlabel('Log(distance km)');
    plt.ylabel('Intensity');

np.arange(magmin,magmax+0.25,0.25)
#magmin,magmax

ymin,ymax = (table.latitude.min(),table.latitude.max())
xmin,xmax = (table.longitude.min(),table.longitude.max())
cx,cy = (np.mean([ymin,ymax]),np.mean([xmin,xmax]))
aspect = (ymax-ymin)/(xmax-xmin)
figwidth = 12
figheight = aspect * figwidth
f = plt.figure(figsize=(figwidth,figheight))
ax = f.add_axes([0.1,0.1,0.8,0.8])
m = Basemap(llcrnrlon=xmin,llcrnrlat=ymin,urcrnrlon=xmax,urcrnrlat=ymax,projection='mill',resolution='h')
m.drawcoastlines();
par = np.arange(np.ceil(ymin),np.floor(ymax)+1,1.0)
mer = np.arange(np.ceil(xmin),np.floor(xmax)+1,1.0)
merdict = m.drawmeridians(mer,labels=[0,0,0,1],fontsize=10,
                          linewidth=0.5,color='gray')
pardict = m.drawparallels(par,labels=[1,0,0,0],fontsize=10,
                          linewidth=0.5,color='gray')
lats = table.latitude.values
lons = table.longitude.values
m.plot(lons,lats,'b^',latlon=True);
plt.title('Station locations for event %s' % maxid);

url = 'http://earthquake.usgs.gov/realtime/product/dyfi/us20006fas/us/1468890673733/cdi_geo.txt'
columns = ['Box','CDI','NumResponses','HypoDistance',
            'Latitude','Longitude','Suspect?','City','State','box2']
table = pd.read_table(url,sep=',',skiprows=[0],names=columns)

table2 = table[table.NumResponses >= 3]
plt.semilogx(table2.HypoDistance,table2.CDI,'b.')

df[df.CDI < 0]



