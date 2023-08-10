#weather underground connection information 
import urllib2
import json

wuAPIkey = 'FFFFFFFFFFFFFFFFF'  #get your free api key at http://www.wunderground.com/weather/api/
zipcode = '90210'            #set your own zip code

url = 'http://api.wunderground.com/api/' + wuAPIkey + '/geolookup/conditions/q/IA/' + zipcode + '.json'

# WSDA Data Location Information
deviceSerial = 'FFFFFFFFFFFFFFFFF' #set your own WSDA serial here to post back to SensorCloud
inSensor     = 'WeatherUnderground'
inChannel1   = 'Temperature'
inChannel2   = 'Humidity'
inChannel3   = 'DewPoint'
inChannel4   = 'WindSpeedMPH'

repo = TimeSeriesRepo(deviceSerial)

try:
    outSeries1 = repo.createTimeSeries(inSensor, inChannel1, 900, 'seconds')
except:
    outSeries1 = repo.getAllTimeSeries(inSensor, inChannel1)[0]
    
try:
    outSeries2 = repo.createTimeSeries(inSensor, inChannel2, 900, 'seconds')
except:
    outSeries2 = repo.getAllTimeSeries(inSensor, inChannel2)[0]
    
try:
    outSeries3 = repo.createTimeSeries(inSensor, inChannel3, 900, 'seconds')
except:
    outSeries3 = repo.getAllTimeSeries(inSensor, inChannel3)[0]
    
try:
    outSeries4 = repo.createTimeSeries(inSensor, inChannel4, 900, 'seconds')
except:
    outSeries4 = repo.getAllTimeSeries(inSensor, inChannel4)[0]

# Getting the data from weather underground
f = urllib2.urlopen(url)
json_string = f.read()
parsed_json = json.loads(json_string)
location = parsed_json['location']['city']
f.close()

# Parsing the json data
rh_s = parsed_json['current_observation']['relative_humidity']
temp_c = parsed_json['current_observation']['temp_c']
dewpoint_c = parsed_json['current_observation']['dewpoint_c']
wind_mph = parsed_json['current_observation']['wind_mph']
time_s = parsed_json['current_observation']['local_epoch']
time_offset_hours = parsed_json['current_observation']['local_tz_offset']

#Turning %RH to float
rh_f = float(rh_s.replace("%",""))

#Turning time in seconds into time in nanoseconds
time_ns = float(time_s) *  NANO_PER_SEC

print time_ns

#creating tupuls (time,value)
outData1 = [(time_ns, temp_c)]
outData2 = [(time_ns, rh_f)]
outData3 = [(time_ns, dewpoint_c)]
outData4 = [(time_ns, wind_mph)]
print 'outData1: ',outData1
print 'outData2: ',outData2
print 'outData3: ',outData3
print 'outData4: ',outData4

#Publishing data to SensorCloud

#outSeries1.push(outData1)
#outSeries1.save()
#outSeries1.tagAsMathengine()

#outSeries2.push(outData2)
#outSeries2.save()
#outSeries2.tagAsMathengine()

#outSeries3.push(outData3)
#outSeries3.save()
#outSeries3.tagAsMathengine()

#outSeries4.push(outData4)
#outSeries4.save()
#outSeries4.tagAsMathengine()

#this will print in the MathEngine output console
print 'this event data is from: ', float(time_s)

#print out the whole json (use for debugging)
parsed_json



