#Check if Required CSV data exists in Working Directory
#if not present data is downloaded from github repo
#Download CSV data from Github
#This allows the notebook to be used outside of the git repo. For.. reasons yet unknown.
from pathlib2 import Path
import wget
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt


combined = Path("m_combined.csv")
if combined.is_file():
    print("m_combined.csv - Exists")
else:
    print("m_combined.csv: DNE -- Downloading from Git")
    url = 'https://raw.githubusercontent.com/nevelo/quake-predict/master/csvData/m_combined.csv'
    fd = wget.download(url)

magCIdat = Path("mag_locationsCI.csv")
if magCIdat.is_file():
    print("mag_locationsCI.csv - Exists")
else:
    print("mag_locationsCI.csv: DNE -- Downloading from Git")
    url = 'https://raw.githubusercontent.com/nevelo/quake-predict/master/csvData/mag_locationsCI.csv'
    fd = wget.download(url)
    
magNCdat = Path("mag_locationsNC.csv")
if magNCdat.is_file():
    print("mag_locationsNC.csv - Exists")
else:
    print("mag_locationsNC.csv: DNE -- Downloading from Git")
    url = 'https://raw.githubusercontent.com/nevelo/quake-predict/master/csvData/mag_locationsNC.csv'
    fd = wget.download(url)

#Open Files
f_combined = open("m_combined.csv")
f_CI = open("mag_locationsCI.csv")
f_NC = open("mag_locationsNC.csv")

mp = Basemap(projection='tmerc',
    width = 120000, height = 900000000,
    lat_0=37.141240, lon_0=-120.046963,
    resolution = 'i', area_thresh = 1000.0,
    llcrnrlon=-126, llcrnrlat=30,
    urcrnrlon=-110, urcrnrlat=44)
 
mp.drawcoastlines()
mp.drawcountries()
mp.fillcontinents()
mp.drawmapboundary()
mp.drawstates(linestyle = 'dotted')

#mp.drawmeridians(np.arange(0, 360, 30))
mp.drawparallels(np.arange(-90, 90, 30))
 
my_map = plt.show()

print("Here is the general region where our data is being pulled from")

#Loading required variables from files
import csv

latsCI, lonsCI, magsCI = [],[],[]
latsNC, lonsNC, magsNC = [],[],[]

def getLatLngMag(to_parse):
    
    lats, lons, mags = [],[],[]
    
    with to_parse as f:
        reader = csv.reader(f)
        
        for row in reader:
            lats.append(float(row[3]))
            lons.append(float(row[4]))
            mags.append(float(row[5]))

    return (lats, lons, mags)

latsCI, lonsCI, magsCI = getLatLngMag(f_CI)
latsNC, lonsNC, magsNC = getLatLngMag(f_NC)

print("Lets check the data")

print("CI Data - first 5 - Size: " + str(len(latsCI)))
print("Lats: ", latsCI[0:10])
print("Lons: ", lonsCI[0:10])
print("Mags: ", magsCI[0:10])
print(" ")
print("NC Data - first 5 - Size: " + str(len(latsNC)))
print("Lats: ", latsNC[0:10])
print("Lons: ", lonsNC[0:10])
print("Mags: ", magsNC[0:10])

#plot histograms of CI
plt.hist(magsCI, color='blue')

plt.xlabel('Readings ID')
plt.ylabel('Magnitudes')
plt.title('Average Magnitudes for CI dataset')

plt.savefig('figs/hist_CI_magnitudes.png')
plt.show()

#plot histograms of NC
plt.hist(magsNC, color='green')

plt.xlabel('Readings ID')
plt.ylabel('Magnitudes')
plt.title('Average Magnitudes for NC dataset')

plt.savefig('figs/hist_NC_magnitudes.png')
plt.show()

CI_start, CI_end = 338000, 340658
NC_start, NC_end = 397004, 397343

plt.plot(magsCI[CI_start:CI_end], color='blue')

plt.xlabel('April 3rd 2010 - Apri 5th 2010')
plt.ylabel('Magnitudes')
plt.title('CI - Baja California Magnitude Readings')

plt.savefig('figs/April-10-2010-MagnitudesCI.png')
plt.show()

plt.plot(magsNC[NC_start:NC_end], color='green')

plt.xlabel('April 3rd 2010 - Apri 5th 2010')
plt.ylabel('Magnitudes')
plt.title('NC - Baja California Magnitude Readings')

plt.savefig('figs/April-10-2010-MagnitudesNC.png')
plt.show()

#Pulling data from m_combined.csv
    
sumMags, meanMags, sumPwr, meanPwr, dayReadings = [],[],[],[],[]
    
with f_combined as f:
    reader = csv.reader(f)
    
    firstLine = True
    for row in reader:
        if firstLine:
            firstLine = False
            continue
        
        sumMags.append(float(row[3]))
        meanMags.append(float(row[4]))
        sumPwr.append(float(row[5]))
        meanPwr.append(float(row[6]))
        dayReadings.append(float(row[9]))

#now we can plot the given days and the general activity giving us a more accurate representation of the aftershocks

plt.plot(dayReadings[7290:7520], color='purple')

plt.xlabel('2010-2-1 to 2010-5-31')
plt.ylabel('Daily Activity')
plt.title('Combined - Baja California Activity')

plt.savefig('figs/April-10-2010-AverageReadings.png')
plt.show()


plt.plot(meanMags[7290:7520], color='purple')

plt.xlabel('2010-2-1 to 2010-5-31')
plt.ylabel('Daily Average Magnitudes')
plt.title('Combined - Baja California Mean Magnitudes')

plt.savefig('figs/April-10-2010-AverageMagnitudes.png')
plt.show()

