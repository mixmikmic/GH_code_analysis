get_ipython().magic('matplotlib inline')
from owslib.sos import SensorObservationService
import pdb
from owslib.etree import etree
import pandas as pd
import datetime as dt
import numpy as np

url = 'http://sdf.ndbc.noaa.gov/sos/server.php?request=GetCapabilities&service=SOS&version=1.0.0'
ndbc = SensorObservationService(url)

# usgs woods hole
# buoy data (single current meter)
url='http://geoport-dev.whoi.edu/thredds/sos/usgs/data2/notebook/1211-AA.cdf'
usgs = SensorObservationService(url)
contents = usgs.contents

usgs.contents

off = usgs.offerings[1]
off.name

off.response_formats

off.observed_properties

off.procedures

# the get observation request below works.  How can we recreate this using OWSLib?
# http://geoport-dev.whoi.edu/thredds/sos/usgs/data2/notebook/1211-A1H.cdf?service=SOS&version=1.0.0&request=GetObservation&responseFormat=text%2Fxml%3Bsubtype%3D%22om%2F1.0.0%22&offering=1211-A1H&observedProperty=u_1205&procedure=urn:ioos:station:gov.usgs:1211-A1H

#pdb.set_trace()
response = usgs.get_observation(offerings=['1211-AA'],
                                 responseFormat='text/xml;subtype="om/1.0.0"',
                                 observedProperties=['http://mmisw.org/ont/cf/parameter/eastward_sea_water_velocity'],
                                 procedure='urn:ioos:station:gov.usgs:1211-AA')

print(response[0:4000])

# usgs woods hole ADCP data
# url='http://geoport-dev.whoi.edu/thredds/sos/usgs/data2/notebook/9111aqd-a.nc'
# adcp = SensorObservationService(url)

root = etree.fromstring(response)

print(root)

# root.findall(".//{%(om)s}Observation" % root.nsmap )
values = root.find(".//{%(swe)s}values" % root.nsmap )

date_value = np.array( [ (dt.datetime.strptime(d,"%Y-%m-%dT%H:%M:%SZ"),float(v))
                      for d,v in [l.split(',') for l in values.text.split()]] )

ts = pd.Series(date_value[:,1],index=date_value[:,0])

ts.plot(figsize=(12,4), grid='on');


start = '1977-01-03T00:00:00Z'
stop = '1977-01-07T00:00:00Z'
response = usgs.get_observation(offerings=['1211-AA'],
                                 responseFormat='text/xml;subtype="om/1.0.0"',
                                 observedProperties=['http://mmisw.org/ont/cf/parameter/eastward_sea_water_velocity'],
                                 procedure='urn:ioos:station:gov.usgs:1211-AA',
                                 eventTime='{}/{}'.format(start,stop))

root = etree.fromstring(response)
date_value = np.array( [ (dt.datetime.strptime(d,"%Y-%m-%dT%H:%M:%SZ"),float(v))
                      for d,v in [l.split(',') for l in values.text.split()]] )
ts = pd.Series(date_value[:,1],index=date_value[:,0])
ts.plot(figsize=(12,4), grid='on');



