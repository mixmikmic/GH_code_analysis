import urllib, os
import pandas as pd

littered = pd.read_csv("C:/Users/Orysya/Desktop/Women_Hackathon/Data_Potential_LongLats/Addresses/littered_samples.csv")
lats = littered['latitude'].tolist()
longs = littered['longtitude'].tolist()

Tests = []
length = len(longs)
for i in range(length):
    Test = str(lats[i])+', '+str(longs[i])
    Tests.append(Test)

myloc = r"C:/Users/Orysya/Desktop/Women_Hackathon/Data_Potential_LongLats/Addresses/littered_area/" #Specify output directory
key = "&key=" + "" #Enter in your API key here

def GetStreet(Add,SaveLoc):
  base = "https://maps.googleapis.com/maps/api/streetview?size=1200x800&location="
  MyUrl = base + Add + key
  fi = Add + ".jpg"
  urllib.urlretrieve(MyUrl, os.path.join(SaveLoc,fi))

for i in Tests:
  GetStreet(Add=i,SaveLoc=myloc)

streets = pd.read_csv("C:/Users/Orysya/Desktop/Women_Hackathon/Data_Potential_LongLats/Addresses/For_Tableau.csv")
streets.head()

lats = streets['lats'].tolist()
longs = streets['longs'].tolist()

Tests = []
length = len(longs)
for i in range(length):
    Test = str(lats[i])+', '+str(longs[i])
    Tests.append(Test)

Tests = Tests[:60000] #Iterate through ~5% of the total data, since API calls are limited and this is a time extensive process

myloc = r"C:/Users/Orysya/Desktop/Women_Hackathon/Data_Potential_LongLats/Addresses/random_SD_addresses2/" #Specify output directory
key = "&key=" + "" #Enter in your API key here

def GetStreet(Add,SaveLoc):
  base = "https://maps.googleapis.com/maps/api/streetview?size=1200x800&location="
  MyUrl = base + Add + key
  fi = Add + ".jpg"
  urllib.urlretrieve(MyUrl, os.path.join(SaveLoc,fi))

for i in Tests:
  GetStreet(Add=i,SaveLoc=myloc)

