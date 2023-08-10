import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_df = pd.read_json("./train.json")

train_df.head(1)

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')
#print_full(train_df.created)
#A brief inspection reveals that the dates when the listings are created fall in 3 months: 4, 5, and 6.

train_df1 = train_df.copy()
train_df1["created"] = pd.to_datetime(train_df1["created"])
train_df1['month'] = train_df1['created'].dt.strftime('%b')

train_df1.head(1)

train_df2 = train_df1.copy()
del train_df2['building_id']
del train_df2['manager_id']
train_df2.head(2)

train_df3 = train_df2.copy()
train_df3['features_number'] = [len(x) for x in train_df3.features]
del train_df3['features']
del train_df3['description']
del train_df3['created']
del train_df3['listing_id']
del train_df3['photos']
train_df3.head(3)

from geopy.geocoders import Nominatim
geolocator = Nominatim(timeout = 199)
location = geolocator.reverse("40.7145, -73.9425").address

print location
print location.split(', ')[-6]
print len(train_df4)

train_df4 = train_df3.copy()
#train_df4['latitude_string'] = [str(x) for x in train_df4.latitude]
#train_df4['longitude_string'] = [str(x) for x in train_df4.longitude]
train_df4['location-ll'] = [str(x) + ', ' + str(y) for x,y in zip(train_df4.latitude, train_df4.longitude)]
del train_df4['latitude']
del train_df4['longitude']

len(train_df4)

suburb = [None] * len(train_df4)
#[geolocator.reverse(x).address for x in train_df4['location-ll']]
suburb[:1000] = [geolocator.reverse(x).address for x in train_df4['location-ll'][:1000]]

suburb_1 = list(suburb)

with open('address.txt', 'w') as f:
    for element in suburb_1:
        if element != None:
            f.write(element.encode('utf8') + '\n')

from geopy.geocoders import GoogleV3
geolocator = GoogleV3(api_key = 'AIzaSyArADAYJx1mXQItgqIVWyv5JEOzi6Qt1ts')

points = train_df4['location-ll'][11000:12000]#Change the numbers here to specify the rows that we want to get address data for.
results = [geolocator.reverse(x, timeout = 10000) for x in points]

results[0][0].address

print train_df4.iloc[10000,]

with open('address10000-11000.txt', 'w') as f:
    for element in results:
        if element != None:
            f.write(element[0].address.encode('utf8') + '\n')
        else:
            f.write('None')





