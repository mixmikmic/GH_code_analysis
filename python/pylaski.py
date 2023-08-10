import requests

baseURL = "https://firemap.sdsc.edu:5443/stations?"
## Previously mentioned Base URL
selectionType = "selection=boundingBox"
## We are using a bounding box to select the station
Lat = 32.88437231
Lon = -117.2414495
## Latitude and Longitude of the center of the bounding box  
selectionParameters = "&minLat=%s&minLon=%s&maxLat=%s&maxLon=%s" % (str(Lat - .075),str(Lon - .075),str(Lat + .075), str(Lon + .075) ) 
## Fill in a string to append to the URL and from a box around the chosen Latitude and Longitude

boundingBoxURL= baseURL + selectionType + selectionParameters
## concatenate strings to form a final URL

print(boundingBoxURL)

## Base URL as mentioned previously
selectionType = "selection=withinRadius"
## Now we will draw a bounding circle around a pair of given Latitude and Longitude points
lat = 32.88437231
lon = -117.2414495
radius = 5
## Latitude and Longitude at the center of the bounding cirlce, along with a 5 km radius to search
selectionParameters = "&lat=%s&lon=%s&radius=%s&observable=temperature" % (str(lat),str(lon),str(radius))
## Fill in parameters to append to the URL

withinRadiusURL= baseURL + selectionType + selectionParameters
## Concatenate the strings to form the final URL

print(withinRadiusURL)

baseURL =  "https://firemap.sdsc.edu:5443/stations/data/latest?"
## Base URL as mentioned previously
selectionType="selection=closestTo"
## Now we will find the closest station to the given point
lat = 32.88437231
lon = -117.2414495
selectionParameters = "&lat=%s&lon=%s&observable=wind_speed" % (str(lat),str(lon))

closestToURL =  baseURL + selectionType + selectionParameters
## Concatenate the strings to form the final URL

print(closestToURL)

originalAmount = len(requests.get(boundingBoxURL).json()['features'])

filterList = ['minimal','light','normal','heavy','maximum']

for filters in filterList:
    filterParameter = "&filtering=%s" % filters
    filterURL = boundingBoxURL + filterParameter

    r_featureAmount = len(requests.get(filterURL).json()['features'])
    print("Using the {} level of filtering results in {} stations to be returned compare to our original {}.".format(filters,r_featureAmount,originalAmount)) 

from pprint import pprint#import pretty print

baseURL = "https://firemap.sdsc.edu:5443/stations/data/latest?"
## Always start with the base of the URL

selectionType="selection=closestTo"
lat = 38.8977
lon = -77.0365
## Latitude and longitude of the White House according to Google
selectionParameters = "&lat=%s&lon=%s&observable=temperature" % (str(lat),str(lon))

infoURL = baseURL + selectionType + selectionParameters

r = requests.get(infoURL)
## Request to GET information from the given URL (Our REST query we built)
r_json = r.json()
## Extract the JSON object from the data returned on our GET request

pprint(r_json)

r_features = r_json['features']
## In the context of the REST Query Interface, 'features' are stations that record various data
## A list of features is returned
r_featureProp = r_features[0]['properties']
## Only recieved ONE feature back, access the first and only feature in the feature list

r_featureDesc = r_featureProp['description']
ID = str(r_featureDesc['wifire_uid'])
## Get the WiFire assigned ID of the station, other IDs are available depending on the feature

distance = r_featureProp['distanceFromLocation']['value']
unit = str(r_featureProp['distanceFromLocation']['units'])
## Get the distance from location and it's respective unit

latitude = r_features[0]['geometry']['coordinates'][1]
longitude = r_features[0]['geometry']['coordinates'][0]
## Get the coordinates of the station's location

printString = "%s is the closest weather station to the White House.\n" % (ID)
printString2 = "It is located at latitude: %.3f and longitude: %.3f, " % (latitude,longitude)
printString3 = "%.3f %s away from the White House" % (distance,unit)

print (printString + printString2 + printString3)

baseURL =  "https://firemap.sdsc.edu:5443/stations/data/latest?"
selectionType = "selection=withinRadius"
lat = 38.8977
lon = -77.0365
radius = 10
## 5 km radius around the White House
selectionParameters = "&lat=%s&lon=%s&radius=%s" % (str(lat),str(lon),str(radius))
## Fill in parameters to append to the URL
observables = "&observable=temperature"

observableURL= baseURL + selectionType + selectionParameters + observables
## Concatenate the strings to form the final URL

r = requests.get(observableURL)
r_json = r.json()
r_features = r_json['features']

r_temp = []
r_distance = []

for feat in r_features:
    featProp = feat['properties']
    r_distance.append( 1 / (featProp['distanceFromLocation']['value']**2) )
    r_temp.append(featProp['temperature']['value'])

weightedEst = 0
    
for i, dist in enumerate(r_distance):
    weightedEst = weightedEst + r_temp[i] * ( dist / sum(r_distance) )

print("Using a weighted average the temperature at the White House is estimated to be: {:.2f} degrees F.".format(weightedEst))

## INSERT QUERY W/ MULTIPLE OBSERVS AND DEMONSTRATE ONE OR MORE VS. ALL ##
baseURL =  "https://firemap.sdsc.edu:5443/stations/data/latest?"
selectionType = "selection=withinRadius"
lat = 38.8977
lon = -77.0365
radius = 10
## 5 km radius around the White House
selectionParameters = "&lat=%s&lon=%s&radius=%s" % (str(lat),str(lon),str(radius))
## Fill in parameters to append to the URL
observables = "&observable=temperature&observable=wind_speed&observable=wind_direction&observable=solar_radiation"

observableURL= baseURL + selectionType + selectionParameters + observables
## Concatenate the strings to form the final URL

r = requests.get(observableURL)
r_json = r.json()
r_features = r_json['features']


print("Our first query has returned {} stations:".format(len(r_features)))

for feat in r_features:
    featProp = feat['properties']
    name = featProp['description']['wifire_uid']
    print("{} measures:".format(name))
    print(str(featProp.keys()))
    print("-------------------------------------------------------------------------------------------------------------------")

print("Using the all parameter:")

observableParameter = "&all=true"
allURL = observableURL + observableParameter

r = requests.get(allURL)
r_json = r.json()
r_features = r_json['features']


print("\nOur filtered query has returned {} stations:".format(len(r_features)))

for feat in r_features:
    featProp = feat['properties']
    name = featProp['description']['wifire_uid']
    print("{} measures:".format(name))
    print(str(featProp.keys()))
    print("-------------------------------------------------------------------------------------------------------------------")

import sys

wrongurl = "https://firemap.sdsc.edu:5443/stations/data/latest?selection=withinRadius&lat=38.8977&lon=-77.0365&radius=10&observable=temperatures&observable=wind_speed&observable=wind_direction&observable=solar_radiatidon&all=true"
r = requests.get(wrongurl)
if r.status_code != 200:
#If the status code is not 'OK'
    print("status code: {}".format(r.status_code))
    if r.status_code == 400:
    #If we have sent a bad request print the message given by the REST API
            print(r.json()['message'])
    #sys.exit(1)

urlBase =  "https://firemap.sdsc.edu:5443/stations?"
urlSelect = "selection=withinRadius"
urlCoords = "&lat=32.6943549&lon=-116.9362629"
urlSelectParam = "&radius=5" 
urlImg = urlBase + urlSelect + urlCoords + urlSelectParam

r = requests.get(urlImg)
r_json = r.json()

if r.status_code != 200:
    print("status code: {}".format(r.status_code))
    sys.exit(1)

r_features = r_json['features']
r_img = []

for feat in r_features:
    r_featProp = feat['properties']
    if ( ('latest-images' in r_featProp) == 1 ):
        print(str(r_featProp.keys()))
        print(str(r_featProp['latest-images'][0][1]['image']))

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np

urlBase =  "https://firemap.sdsc.edu:5443/stations/data/latest?"
urlSelect = "selection=withinRadius"
lat = "32.88437231"
lon = "-117.2414495"
urlCoords = "&lat=%s&lon=%s" % (lat,lon)
urlSelectParam = "&radius=15"
urlObserv = "&observable=temperature&observable=wind_speed&observable=wind_direction&all=true"

urlFinal = urlBase + urlSelect + urlCoords + urlSelectParam + urlObserv

r = requests.get(urlFinal)
r_json = r.json()

print(urlFinal)

if r.status_code != 200:
	print("status code: {}".format(r.status_code))
	sys.exit(1)

r_features = r_json['features']

r_lat = []
r_lon = []
r_wind_speed = []
r_wind_direction = []
r_temperature = []
r_unit = []

for feat in r_features:
    r_featProp = feat['properties']
    keys = str(r_featProp.keys())
    name = str(r_featProp['description']['wifire_uid'])

    r_lat.append(feat['geometry']['coordinates'][1])
    r_lon.append(feat['geometry']['coordinates'][0])
    
    r_wind_speed.append(r_featProp['wind_speed']['value'])
    r_wind_direction.append(r_featProp['wind_direction']['value'])
    r_temperature.append(r_featProp['temperature']['value'])
    r_unit = r_featProp['temperature']['units']

plot2 = plt.figure()
ax = plt.gca()
ax.get_xaxis().get_major_formatter().set_useOffset(False)
v_x = r_wind_speed * np.cos(r_wind_direction)
v_y = r_wind_speed * np.sin(r_wind_direction)
plt.quiver(r_lon,r_lat,v_x,v_y,r_temperature, headlength = 4,zorder =1 )
colorbarLabel = "temperature(%s)" % r_unit
plt.colorbar(label=colorbarLabel)
plt.title('lat=32.6616377&lon=-117.0831703')
plt.xlabel("lon (deg.N)")
plt.ylabel("lat (deg.W)")
plt.show()

import arrow

baseURL = "https://firemap.sdsc.edu:5443/stations/data?"
## Always start with the base of the URL

selectionType="selection=closestTo"
#15.87, -97.08
lat = 82.517778 
lon = -62.280556

## Latitude and longitude of the White House according to Google
selectionParameters = "&lat=%s&lon=%s" % (str(lat),str(lon))
observables = "&observable=temperature"

to = arrow.Arrow.now()
frm = to.replace(days=-1)
urlDateTime = "&from=%s&to=%s" % ( str(frm) , str(to) )

urlPlot = baseURL + selectionType + selectionParameters + observables + urlDateTime
print(urlPlot)

r = requests.get(urlPlot)
## Request to GET information from the given URL (Our REST query we built)
r_json = r.json()
## Extract the JSON object from the data returned on our GET request

rTemperature = r_json['features'][0]['properties']['temperature']
rTime = r_json['features'][0]['properties']['timestamp']

rTimeMins = []
for i, val in enumerate(rTime):
	rTimeMins.append( (arrow.get(rTime[i]).timestamp - arrow.get(rTime[0]).timestamp) / 60 / 60 )

plt.plot(rTimeMins,rTemperature,'-og',label='Actual')
plt.xlabel("time (hours)")
plt.ylabel("temperature (F)")
minFive = int(min(rTemperature)) - (int(min(rTemperature))%5)
maxFive = (int(max(rTemperature)) +5) - ( (int(max(rTemperature)) +5) % 5 )
plt.yticks(np.arange(minFive, maxFive+1, (maxFive - minFive)/2))
plt.gca().yaxis.grid(True)
title = "Temperature at Alert Airport, NU\n(Past 24 Hours)"
plt.title(title)
plt.show()

import pandas as pd

baseURL = 'https://firemap.sdsc.edu:5443/'
datatypeURL = 'stations/data?'
selectURL = 'selection=withinRadius'
selectParam = '&lat=32.8842436&lon=-117.2398167&radius=3'
observURL = '&observable=temperature&observable=wind_speed&observable=relative_humidity'
dateTo = arrow.now()
dateFrom = dateTo.replace(days=-1)
dateURL = '&from=%s&to=%s' % (dateFrom,dateTo)

urlPast = baseURL + datatypeURL + selectURL + selectParam + observURL + dateURL

r = requests.get(urlPast)
r_json = r.json()

if r.status_code != 200:
    print("status code: {}".format(r.status_code))
    sys.exit(1)

r_features = r_json['features']
observKeys = ['relative_humidity','temperature','wind_speed']
for feat in r_features:
    r_featProp = feat['properties']
    name = str(r_featProp['description']['wifire_uid'])
    df = pd.DataFrame()
    print("\t\t{}:\n--------------------------------------------------".format(name))

    for prop in r_featProp:
        if prop in observKeys:
            df[prop] = r_featProp[prop]
    print(df.describe())
    print("\n--------------------------------------------------")

urlBase =  "https://firemap.sdsc.edu:5443/"
urlQuery = "forecast?"
urlForecastParam = "hrrrx=true&"
urlSelect = "selection=withinRadius"
urlCoords = "&lat=33.100492&lon=-116.3013267"
urlSelectParam = "&radius=2"


urlNWS = urlBase + urlQuery + 'hrrrx=false&' + urlSelect + urlCoords + urlSelectParam

print(urlNWS)

NWS = requests.get(urlNWS)
##########################################################
if NWS.status_code != 200:
    print("status code: {}".format(NWS.status_code))
    if 'message' in NWS.json().keys():
        print(NWS.json()['message'] )
    sys.exit(1)
##########################################################
NWSData = NWS.json()

r_featProp = NWSData['properties']
r_windTimes = []
winds = 0
for i, val in enumerate(r_featProp['wind_speed']):
    if (r_featProp['wind_speed'][i] >= 11.2) and ( (10<=r_featProp['wind_direction'][i]) and (r_featProp['wind_direction'][i] <= 110) ) and (r_featProp['relative_humidity'][i] >= 25):
        winds = winds + 1
        r_windTimes.append(r_featProp['timestamp'][i])   
        print("A Santa Ana Wind is predicted to occur at: {} {}".format(r_featProp['timestamp'][i][0:10],r_featProp['timestamp'][i][11:-6]))

urlHRRRX = urlBase + urlQuery + urlForecastParam + urlSelect + urlCoords + urlSelectParam

HRRRX = requests.get(urlHRRRX)
print(urlHRRRX)
##########################################################
if HRRRX.status_code != 200:
	print("status code: {}".format(HRRRX.status_code))
	if 'message' in HRRRX.json().keys():
		print( HRRRX.json()['message'] )
	sys.exit(1)
##########################################################
HRRRXData = HRRRX.json()

NWSTime = []
for i, val in enumerate(NWSData['properties']['timestamp']):
    NWSTime.append( (arrow.get(NWSData['properties']['timestamp'][i]).timestamp - arrow.get(NWSData['properties']['timestamp'][0]).timestamp) / 60 )
    if ( ( (arrow.get(NWSData['properties']['timestamp'][i]).timestamp - arrow.get(HRRRXData['properties']['timestamp'][-1]).timestamp) / 60 ) >= 0 ):
        break

HRRRXTime = []
for i, val in enumerate(HRRRXData['properties']['timestamp']):
    HRRRXTime.append( (arrow.get(HRRRXData['properties']['timestamp'][i]).timestamp - arrow.get(HRRRXData['properties']['timestamp'][0]).timestamp) / 60 )
    
NWSLine = plt.plot(NWSTime,NWSData['properties']['temperature'][0:len(NWSTime)],'-or',label='NWS' )
HRRRXLine = plt.plot(HRRRXTime,HRRRXData['properties']['temperature'],'-og',label='HRRRX')
plt.xlabel("time (minutes)")
plt.ylabel("temperature (F)")
plt.legend()
title = "HRRRX Forecast vs. NWS Forecast\nin Anza Borrego Desert State Park"
plt.title(title)
plt.show()



