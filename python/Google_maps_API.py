import googlemaps
from datetime import datetime

gmaps = googlemaps.Client(key='somesecretkeyhere')

# Geocoding an address
geocode_result = gmaps.geocode('1600 Amphitheatre Parkway, Mountain View, CA')

type(geocode_result)

from pprint import pprint
pprint(geocode_result)

# Look up an address with reverse geocoding
reverse_geocode_result = gmaps.reverse_geocode((40.714224, -73.961452))

pprint(reverse_geocode_result)

# Request directions via public transit
now = datetime.now()
directions_result = gmaps.directions("Yerevan, Armenia","Dilijan, Armenia", departure_time=now)

pprint(directions_result)

for i in directions_result:
    legs = i["legs"]
    for leg in legs:
        dist = leg["distance"]
        print dist

type(dist)

loc1 = raw_input("Please, provide the start location: ")+", Armenia"
loc2 = raw_input("Please, provide the end location: ")+", Armenia"
directions_result = gmaps.directions(loc1,loc2, departure_time=datetime.now())
for i in directions_result:
    legs = i["legs"]
    for leg in legs:
        dist = leg["distance"]
print("\n")
print("The distance between "+loc1+" and "+loc2+" is "+dist["text"])

