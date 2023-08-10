import httplib
import base64
import string
import json
import urllib
import time
import calendar
from datetime import datetime

# SensorCloud Device, Sensor, and Channel Information
DEVICE_SERIAL = 'YOUR_SENSORCLOUD_DEVICE_SERIAL'
SENSOR_ID = '30861'
CHANNEL_NAME = 'ch4'

#LIFX API variables
LIFX_HOST = "api.lifx.com"
LIFX_TOKEN = "YOUR_LIFX_API_TOKEN"

#The temperature (in degrees celsius) to change the lights at
COFFEE_FRESH_MIN = 60
COFFEE_OK_MIN = 50

def getDataSeries(serial, sensor, channel, rate, rateType, start=None):
    """
    Function: getDataSeries
        Gets the Data Series object from SensorCloud for the given information.
        
    Parameters:
        serial - The serial of the device to get data from.
        sensor - The sensor name to get data from.
        channel - The channel name to get data from.
        rate - The sample rate of the data to retrieve.
        rateType - The sample rate type ('hertz' or 'seconds') of the data to retrieve.
        start - The start time of the data to retrieve (optional).
        
    Returns:
        The SensorCloud time-series for the given parameters.
    """
    
    #get the actual data for the device/sensor/channel
    repo = TimeSeriesRepo(serial)
    dataSeries  = repo.getAllTimeSeries(sensor, channel, startTime=start, sampleRate=rate, sampleRateType=rateType)
    
    return dataSeries

def chooseLightColor(val):
    """
    Function: chooseLightColor
        Determines the Lifx color string we should use to send to the light.
        
    Parameters:
        val - The channel value (in degrees celsius).
        
    Returns:
        The Lifx color string to pass to the bulb.
    """

    #if the coffee is fresh
    if val >= COFFEE_FRESH_MIN:
        print "Coffee is Fresh!"
        return "green brightness:100%"
    
    #if the coffee is just ok
    elif val >= COFFEE_OK_MIN:
        print "Coffee is ok"
        return "#003E52 brightness:100%"
        
    #if the coffee is old
    else:
        print "Coffee is kind of old..."
        return "off"

def breatheLifxLight(color, period, cycles):
    """
    Function: breatheLifxLight
        Uses the LIFX API to "breathe" the lightbulbs.
        
    Parameters:
        color - The color to breathe.
        period - The time (in seconds) for one cycle of the effect.
        cycle - The number of time to repeat the effect.
    """
    
    #create the connection
    conn = httplib.HTTPSConnection(LIFX_HOST)
    
    data = urllib.urlencode({'color': 'white', 'from_color': color, 'period': period, 'cycles': cycles, 'power_on': 'true', 'persist': 'true'});
    conn.putrequest("POST", "/v1/lights/group:Coffee/effects/breathe", data)
        
    print "LIFX: Breathing " + color
    
    conn.putheader("Host", LIFX_HOST);
    conn.putheader("Authorization", "Bearer %s" % LIFX_TOKEN)
    conn.putheader("Content-type", "application/x-www-form-urlencoded; charset=UTF-8")
    conn.putheader("Content-length", str(len(data)))
    conn.endheaders()
    
    #send the request and get the response
    conn.send(data)
    r = conn.getresponse()
    print r.status, r.reason
    print r.read()

def setLifxLight(color):
    """
    Function: setLifxLight
        Uses the LIFX API to set the color of the lightbulbs.
        
    Parameters:
        color - The color to change the bulb to.
    """
    
    #create the connection
    conn = httplib.HTTPSConnection(LIFX_HOST)
    
    if color != "off":
        data = urllib.urlencode({'color': color})
        conn.putrequest("PUT", "/v1/lights/group:Coffee/state", data)
        print "LIFX: Setting color to " + color
        
    else:
        data = urllib.urlencode({'power': "off"})
        conn.putrequest("PUT", "/v1/lights/group:Coffee/state", data)
        print "LIFX: Turning off"
    
    conn.putheader("Host", LIFX_HOST);
    conn.putheader("Authorization", "Bearer %s" % LIFX_TOKEN)
    conn.putheader("Content-type", "application/x-www-form-urlencoded; charset=UTF-8")
    conn.putheader("Content-length", str(len(data)))
    conn.endheaders()
    
    #send the request and get the response
    conn.send(data)
    r = conn.getresponse()
    print r.status, r.reason
    print r.read()

# The sample rate of the SensorCloud data to find
sampleRate = 2
sampleRateType = "seconds"

lastColor = ""
lastVal = 0
lastBrewTime = 0

while True:
    
    # get the data series for the Device/Node/Channel (for all time)
    # Note: getDataSeries doesn't load all the data into memory, so this operation is small
    tempSeries = getDataSeries(DEVICE_SERIAL, SENSOR_ID, CHANNEL_NAME, sampleRate, sampleRateType)
    
    # find the last timestamp in the data
    startTime = tempSeries[0].getEndTimeStamp()
    
    # get the data series again, this time asking for only the last 1 second of data
    # since we only care about the most recent data, which is more efficient
    allSeries = getDataSeries(DEVICE_SERIAL, SENSOR_ID, CHANNEL_NAME, sampleRate, sampleRateType, startTime - 1000)
    
    # get the first series (gives you one per sample rate)
    data = allSeries[0]
    
    # if we found data
    if data != None:
        
        # look at the last point in the data
        point = data[-1]
        
        #get the timestamp and data value of the point
        ts = point[0]
        val = point[1]
    
        print "System time:", datetime.now().strftime("%x %X")
        print "Last Point = [%s, %s]" % point
        
        color = chooseLightColor(val)
        
        #attempt to detect a new brew cycle (temperature rose by at least 2 degrees)
        if lastVal != 0 and val > (lastVal + 2):
            
            tempBrewTime = calendar.timegm(time.gmtime())
            
            #assume brews can't occur within 2 minutes of each other
            if (tempBrewTime - lastBrewTime) > 120:
                print "New Brew Cycle!"
                lastBrewTime = calendar.timegm(time.gmtime())
                breatheLifxLight(color, 2.0, 30.0)
                lastColor = color
                
        else:
            # If the color has changed
            if color != lastColor:
                setLifxLight(color)
                lastColor = color
            
        lastVal = val
        
    print ""
        
    time.sleep(15)

