import googlemaps
from datetime import datetime
import simplejson
import urllib, json, time
import pandas as pd
import math
import dateutil 
import datetime as dt
from datetime import datetime, timedelta

apikey = 'XXXXXXXXXXXXXXXXXXXXXX'

df = pd.read_csv('final_top_bikes.csv')

df.info()

def pmam(x):
    x = str(x)
    #x = (':'.join(a+b for a,b in zip(x[::2], x[1::2])))
    if x == 'NaN':
        pass
    try:
        x = str(x[:2] + ':' + x[2:])
        date = dateutil.parser.parse(x)
        return str(date.strftime('%d/%m/%Y %H:%M %p'))
    except:
        return 'NaN'

print(pmam('2017-09-09 07:00:03'))

df['Timestamp index'] = df['Timestamp +2'].apply(pmam)
df['Timestamp index'] = df['Timestamp index'].apply(lambda x: 
                                    dt.datetime.strptime(x,'%d/%m/%Y %H:%M %p'))
df.index = df['Timestamp index']

import urllib, json, time
import pandas as pd

def google(lato, lono, latd, lond):

    url = """https://maps.googleapis.com/maps/api/distancematrix/json?origins=%s,%s"""%(lato, lono)+      """&destinations=%s,%s&mode=bicycling&language=en-EN&sensor=false&key="""+apikey%(latd, lond)
    #print(url)
    #CHANGE THIS FOR PYTHON 3.X TO urllib.request.urlopen(url)...
    response = urllib.request.urlopen(url).read().decode('utf8')

    #Wait a second so you don't overwhelm the API if doing lots of calls
    time.sleep(1)

    obj = json.loads(response)
    try:
        s =   obj['rows'][0]['elements'][0]['duration']['value']
        m = (obj['rows'][0]['elements'][0]['distance']['value'])
        return s, m
    except IndexError:
        #something went wrong, the result was not found
        print (url)
        #return the error code
        return obj['Status'], obj['Status']

def ApplyGoogle(row):
    lato, lono = row['Lat'], row['Long']
    latd, lond = row['newLat'], row['newLong']
    return google(lato, lono, latd, lond)

journeydf['Seconds'], journeydf['Metres'] = zip(*journeydf.apply(ApplyGoogle, axis = 1))

journeydf.to_csv('data.csv')

