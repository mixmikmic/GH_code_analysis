import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

plt.style.use('ggplot')

# earthquakes dataframe
earthquakes = pd.read_csv('world_eq.csv')
earthquakes.head(10)

len(earthquakes.index)

earthquakes = earthquakes[["Date", "Time", "Latitude","Longitude","Magnitude", "Depth"]]
earthquakes.head()

# tsunamis dataframe
tsunamis = pd.read_excel('tsevent.xlsx')
tsunamis.head()

from mpl_toolkits.basemap import Basemap
from matplotlib.colors import rgb2hex
from matplotlib.patches import Polygon

for i in range(0, len(tsunamis.columns.values)):
    tsunamis.columns.values[i] = str(tsunamis.columns.values[i])

# delete unnecessary columns
tsunamis.drop(tsunamis.columns[[range(16,46)]], inplace = True, axis = 1)
tsunamis = tsunamis[["ID", "YEAR", "MONTH", "DAY", "HOUR", "MINUTE", "COUNTRY", "STATE", "LOCATION_NAME", "LATITUDE", "LONGITUDE"]]
tsunamis.head()

# Drop N/A lon/lat values for tsunami
# I filtered with longitude because if longitude has N/A, corresponding latitude also has it
tsu = tsunamis.loc[np.isnan(tsunamis['LONGITUDE']) == False]
tsu.head()

recenttsu = tsu.loc[tsunamis['YEAR'] > 1964]
recenttsu.head()

len(recenttsu.index)

# draw world map

plt.figure(figsize=(15,10))
displaymap = Basemap(llcrnrlon=-180,llcrnrlat=-90,urcrnrlon=180,urcrnrlat=90)

displaymap.drawmapboundary()
displaymap.drawcountries()
displaymap.drawcoastlines()

# Convert longitudes and latitudes to list of floats
longitude = earthquakes[['Longitude']].values.tolist()
for i in range(0, len(longitude)):
    longitude[i] = float(longitude[i][0])
latitude = earthquakes[['Latitude']].values.tolist()
for i in range(0, len(latitude)):
    latitude[i] = float(latitude[i][0])
tlongitude = recenttsu[[u'LONGITUDE']].values.tolist()
for i in range(0, len(tlongitude)):
    tlongitude[i] = float(tlongitude[i][0])
tlatitude = recenttsu[[u'LATITUDE']].values.tolist()
for i in range(0, len(tlatitude)):
    tlatitude[i] = float(tlatitude[i][0])

lons,lats = displaymap(longitude, latitude)
tlons, tlats = displaymap(tlongitude, tlatitude)
displaymap.plot(lons, lats, 'bo', color = "blue")
displaymap.plot(tlons, tlats, 'bo', color = "red")

plt.title("Earthquakes and Tsunamis around the World from `1965-2017")
plt.show()

dates = earthquakes[['Date']].values.tolist()
years = []
months = []
days = []
for i in range(0, len(dates)):
    dates[i] = dates[i][0].split("/")
    try:
        years.append(dates[i][2])
    except IndexError:
        years.append('NaN')
    try:
        months.append(dates[i][0])
    except IndexError:
        months.append('NaN')
    try:
        days.append(dates[i][1])
    except IndexError:
        days.append('NaN')

idlist = []
for i in range(0, len(earthquakes.index)):
    idlist.append(i)

earthquakes['Year'] = years
earthquakes['Month'] = months
earthquakes['Days'] = days
earthquakes['ID'] = idlist

earthquakes.head()

float(len(recenttsu.index))/float(len(earthquakes.index))

eq2012 = earthquakes.loc[(earthquakes['Year'] == '2012')]
tsu2012 = tsu.loc[tsu[u'YEAR'] == 2012]

tsu2012

print len(tsu2012), len(eq2012)

tsu2012.loc[tsu2012[u'MONTH'] == 2]

eq2012.loc[(eq2012['Month'] == '2') & (eq2012['Days'] == '2')]

earthquakes.loc[earthquakes['ID'] == 21144]

tsu2012.loc[tsu2012[u'MONTH'] == 3]

eq2012.loc[(eq2012['Month'] == '3') & ((eq2012['Days'] == '14') | (eq2012['Days'] == '20'))]

earthquakes.loc[(earthquakes['ID'] == 21192) | (earthquakes['ID'] == 21203)]

tsu2012.loc[tsu2012[u'MONTH'] == 4]

eq2012.loc[(eq2012['Month'] == '4') & ((eq2012['Days'] == '11') | (eq2012['Days'] == '14'))]

earthquakes.loc[(earthquakes['ID'] == 21219) | (earthquakes['ID'] == 21224) | (earthquakes['ID'] == 21238)]

tsu2012.loc[tsu2012[u'MONTH'] == 7]

eq2012.loc[(eq2012['Month'] == '7') & (eq2012['Days'] == '15')]

tsu2012.loc[tsu2012[u'MONTH'] == 8]

eq2012.loc[(eq2012['Month'] == '8') & ((eq2012['Days'] == '27') | (eq2012['Days'] == '31'))]

earthquakes.loc[(earthquakes['ID'] == 21405) | (earthquakes['ID'] == 21411)]

tsu2012.loc[tsu2012[u'MONTH'] == 9]

eq2012.loc[(eq2012['Month'] == '9') & (eq2012['Days'] == '5')]

earthquakes.loc[(earthquakes['ID'] == 21418)]

tsu2012.loc[tsu2012[u'MONTH'] == 10]

eq2012.loc[(eq2012['Month'] == '10') & (eq2012['Days'] == '28')]

earthquakes.loc[(earthquakes['ID'] == 21477)]

tsu2012.loc[tsu2012[u'MONTH'] == 11]

eq2012.loc[(eq2012['Month'] == '11') & (eq2012['Days'] == '7')]

earthquakes.loc[(earthquakes['ID'] == 21493)]

tsu2012.loc[tsu2012[u'MONTH'] == 12]

eq2012.loc[(eq2012['Month'] == '12') & ((eq2012['Days'] == '7') | (eq2012['Days'] == '28'))]

earthquakes.loc[(earthquakes['ID'] == 21530)]

eqtsu2012 = earthquakes.loc[(earthquakes['ID'] == 21144) | (earthquakes['ID'] == 21192) | (earthquakes['ID'] == 21203) | 
                (earthquakes['ID'] == 21405) | (earthquakes['ID'] == 21219) | (earthquakes['ID'] == 21224) | 
                (earthquakes['ID'] == 21238) | (earthquakes['ID'] == 21405) | (earthquakes['ID'] == 21411) | 
                (earthquakes['ID'] == 21418) | (earthquakes['ID'] == 21477) | (earthquakes['ID'] == 21493) | 
                (earthquakes['ID'] == 21530)]

eqtsu2012

print float(len(eqtsu2012))/float(len(tsu2012)), float(len(eqtsu2012))/float(len(eq2012))

plt.figure(figsize=(15,10))
displaymap2012 = Basemap(llcrnrlon=-180,llcrnrlat=-90,urcrnrlon=180,urcrnrlat=90)
displaymap2012.drawmapboundary()
displaymap2012.drawcountries()
displaymap2012.drawcoastlines()
longitude2012 = eqtsu2012[['Longitude']].values.tolist()
for i in range(0, len(longitude2012)):
    longitude2012[i] = float(longitude2012[i][0])
latitude2012 = eqtsu2012[['Latitude']].values.tolist()
for i in range(0, len(latitude2012)):
    latitude2012[i] = float(latitude2012[i][0])
lons2012,lats2012 = displaymap(longitude2012, latitude2012)
displaymap2012.plot(lons2012, lats2012, 'bo', color = "blue")

plt.title("Earthquakes that Caused Tsunamis in 2012")
plt.show()

min2012 = eqtsu2012['Magnitude'].min()
max2012 = eqtsu2012['Magnitude'].max()
print min2012, max2012

plt.figure(figsize=(10,10))
plt.hist(eqtsu2012['Magnitude'], bins = 5, alpha = 0.4)
plt.xlabel('Magnitude')
plt.ylabel('Frequency')
plt.title("Frequencies of Earthquakes that Caused Tsunamis in 2012")
plt.show()

eq1997 = earthquakes.loc[(earthquakes['Year'] == '1997')]
tsu1997 = tsu.loc[tsu[u'YEAR'] == 1997]

tsu1997

print len(tsu1997.index), len(eq1997.index)

tsu1997.loc[tsu1997[u'MONTH'] == 4]

eq1997.loc[(eq1997['Month'] == '4') & ((eq1997['Days'] == '10') | (eq1997['Days'] == '21'))]

eq1997.loc[(eq1997['ID'] == 13496)]

tsu1997.loc[tsu1997[u'MONTH'] == 7]

eq1997.loc[(eq1997['Month'] == '7') & (eq1997['Days'] == '9')]

eq1997.loc[(eq1997['ID'] == 13600)]

tsu1997.loc[tsu1997[u'MONTH'] == 9]

eq1997.loc[(eq1997['Month'] == '9') & (eq1997['Days'] == '30')]

eq1997.loc[(eq1997['ID'] == 13688)]

tsu1997.loc[tsu1997[u'MONTH'] == 10]

eq1997.loc[(eq1997['Month'] == '10') & (eq1997['Days'] == '14')]

eq1997.loc[(eq1997['ID'] == 13711)]

tsu1997.loc[tsu1997[u'MONTH'] == 12]

eq1997.loc[(eq1997['Month'] == '12') & ((eq1997['Days'] == '5') | (eq1997['Days'] == '14') | (eq1997['Days'] == '26'))]

eq1997.loc[(eq1997['ID'] == 13785)]

eqtsu1997 = earthquakes.loc[(earthquakes['ID'] == 13469) | (earthquakes['ID'] == 13600) | (earthquakes['ID'] == 13688) | 
                (earthquakes['ID'] == 13711) | (earthquakes['ID'] == 23785)]

eqtsu1997

print float(len(eqtsu1997))/float(len(tsu1997)), float(len(eqtsu1997))/float(len(eq1997))

plt.figure(figsize=(15,10))
displaymap1997 = Basemap(llcrnrlon=-180,llcrnrlat=-90,urcrnrlon=180,urcrnrlat=90)
displaymap1997.drawmapboundary()
displaymap1997.drawcountries()
displaymap1997.drawcoastlines()
longitude1997 = eqtsu1997[['Longitude']].values.tolist()
for i in range(0, len(longitude1997)):
    longitude1997[i] = float(longitude1997[i][0])
latitude1997 = eqtsu1997[['Latitude']].values.tolist()
for i in range(0, len(latitude1997)):
    latitude1997[i] = float(latitude1997[i][0])
lons1997,lats1997 = displaymap(longitude1997, latitude1997)
displaymap1997.plot(lons1997, lats1997, 'bo', color = "blue")

plt.title("Earthquakes that Caused Tsunamis in 1997")
plt.show()

min1997 = eqtsu1997['Magnitude'].min()
max1997 = eqtsu1997['Magnitude'].max()
print min1997, max1997

eqcom = earthquakes.loc[(earthquakes['Year'] == '1997') | (earthquakes['Year'] == '2012')]
tsucom = tsu.loc[(tsu[u'YEAR'] == 1997) | (tsu[u'YEAR'] == 2012)]
frames = [eqtsu1997, eqtsu2012]
eqtsucom = pd.concat(frames)

eqtsucom

print float(len(eqtsucom))/float(len(tsucom)), float(len(eqtsucom))/float(len(eqcom))

plt.figure(figsize=(15,10))
displaymapcom = Basemap(llcrnrlon=-180,llcrnrlat=-90,urcrnrlon=180,urcrnrlat=90)
displaymapcom.drawmapboundary()
displaymapcom.drawcountries()
displaymapcom.drawcoastlines()
longitudecom = eqtsucom[['Longitude']].values.tolist()
for i in range(0, len(longitudecom)):
    longitudecom[i] = float(longitudecom[i][0])
latitudecom = eqtsucom[['Latitude']].values.tolist()
for i in range(0, len(latitudecom)):
    latitudecom[i] = float(latitudecom[i][0])
lonscom,latscom = displaymap(longitudecom, latitudecom)
displaymapcom.plot(lonscom, latscom, 'bo', color = "blue")

plt.title("Earthquakes that Caused Tsunamis in Both Years")
plt.show()

mincom = eqtsucom['Magnitude'].min()
maxcom = eqtsucom['Magnitude'].max()
print mincom, maxcom

plt.figure(figsize=(10,10))
plt.hist(eqtsucom['Magnitude'], bins = 5, alpha = 0.4)
plt.xlabel('Magnitude')
plt.ylabel('Frequency')
plt.title("Frequencies of Earthquakes that Caused Tsunamis in Combined Dataset")
plt.show()

