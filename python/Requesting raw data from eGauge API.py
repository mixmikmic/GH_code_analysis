host = 'http://egauge{}.egaug.es/cgi-bin/egauge-show?'

sensor_id = 34111 #ACC House 1 ; Purpose ID 136

import arrow
iso_start_datetimez = '2017-09-13T15:19:00-10'
iso_end_datetimez = '2017-09-13T15:25:00-10'

start_timestamp = arrow.get(iso_start_datetimez).timestamp
end_timestamp = arrow.get(iso_end_datetimez).timestamp

minutes = 'm'

output_csv = 'c'

delta_compression = 'C'

import requests

host = host.format(sensor_id) + '&' + minutes + '&' + output_csv + '&' + delta_compression 
time_window = {'t': start_timestamp, 'f': end_timestamp}

print(host)

r = requests.get(host,params=time_window)
print(r)

import pandas as pd
from io import StringIO #To parse the String as a .csv file
df = pd.read_csv(StringIO(r.text))
df

iso_start_datetimez = '2017-09-13T15:25:00-10' 
iso_end_datetimez = '2017-09-13T21:25:00-10' #hour (after T) was adjusted from 15 to 21. 

start_timestamp = arrow.get(iso_start_datetimez).timestamp
end_timestamp = arrow.get(iso_end_datetimez).timestamp

hour = 'h' #previously minutes = 'm'

host = host + '&' + hour + '&' + output_csv + '&' + delta_compression 
time_window = {'t': start_timestamp, 'f': end_timestamp}
r = requests.get(host,params=time_window)
df = pd.read_csv(StringIO(r.text))
df

host = 'http://egauge{}.egaug.es/cgi-bin/egauge-show?'
host = host.format(sensor_id) + '&' + hour + '&' + output_csv ## delta_compression variable removed
r = requests.get(host,params=time_window)
string_csv_file = r.text    
df = pd.read_csv(StringIO(r.text))
df

# As originally used in the first request by the notebook

iso_start_datetimez = '2017-09-13T15:19:00-10'
iso_end_datetimez = '2017-09-13T15:25:00-10'

start_timestamp = arrow.get(iso_start_datetimez).timestamp
end_timestamp = arrow.get(iso_end_datetimez).timestamp

time_window = {'t': start_timestamp, 'f': end_timestamp}



host = 'http://egauge{}.egaug.es/cgi-bin/egauge-show?'
host = host.format(sensor_id) + '&' + minutes + '&' + output_csv + '&' + delta_compression 
r = requests.get(host,params=time_window)
string_csv_file = r.text    
df = pd.read_csv(StringIO(r.text))
df

host = 'http://egauge{}.egaug.es/cgi-bin/egauge-show?'
host = host.format(sensor_id) + '&' + minutes + '&' + output_csv # No delta-compression
r = requests.get(host,params=time_window)
string_csv_file = r.text    
df = pd.read_csv(StringIO(r.text))
df

import xml.dom.minidom

iso_start_datetimez = '2017-09-13T15:19:00-10'
iso_end_datetimez = '2017-09-13T15:20:00-10'

start_timestamp = arrow.get(iso_start_datetimez).timestamp
end_timestamp = arrow.get(iso_end_datetimez).timestamp

time_window = {'t': start_timestamp, 'f': end_timestamp}


host = 'http://egauge{}.egaug.es/cgi-bin/egauge-show?'
host = host.format(sensor_id) + '&' + minutes
r = requests.get(host,params=time_window)
string_xml_file = r.text    

import xml.etree.ElementTree as etree
from xml.dom import minidom

x = etree.fromstring(string_xml_file)
def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = etree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")
print(prettify(x))

