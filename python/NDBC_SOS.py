get_ipython().magic('matplotlib inline')

from datetime import datetime
import cStringIO
import pandas as pd
from owslib.sos import SensorObservationService

# pick a buoy, any buoy
#sta_id='44066'  # texas tower
sta_id='44013'  # boston buoy

# pick a start & stop time
start = '2013-06-12T00:00:00Z'
stop = '2013-06-14T00:00:00Z'

from IPython.core.display import HTML
HTML('<iframe src=http://www.ndbc.noaa.gov/station_page.php?station=%s width=950 height=400></iframe>' % sta_id)

ndbc=SensorObservationService('http://sdf.ndbc.noaa.gov/sos/server.php?request=GetCapabilities&service=SOS')

id=ndbc.identification
id.title

contents = ndbc.contents
network = contents['network-all']
network.description

id.title

rfs = network.response_formats

print '\n'.join(rfs)

station = contents['station-%s' % sta_id]    

station.name

station.description

getob = ndbc.get_operation_by_name('getobservation')

getob.parameters

# issue the SOS get_obs request
response = ndbc.get_observation(offerings=['urn:ioos:station:wmo:%s' % sta_id],
                                 responseFormat='text/csv',
                                 observedProperties=['winds'],
                                 eventTime='%s/%s' % (start,stop))
                                 

response

df2 = pd.read_csv(cStringIO.StringIO(response.strip()),index_col='date_time',parse_dates=True)  # skip the units row 

df2.head()

df2[['wind_speed_of_gust (m/s)','wind_speed (m/s)']].plot(figsize=(12,4),title=station.description,legend=True)



