from json import loads, dumps
import pandas as pd
import numpy as np
# EXTRACT
with open("all-flights.json") as f1:
    sched_flights = pd.DataFrame(loads(f1.read())["AirlineFlightSchedulesResult"]["data"])

with open("historical_flights.json") as f2:
    hist_flights = pd.DataFrame(loads(f2.read())).T

with open("airports-new.json") as f3:
    airports = loads(f3.read())
    ap = pd.DataFrame(airports).T

# TRANSFORM
import arrow

sched_flights['arrowdate'] = sched_flights.departuretime.apply(arrow.get)
sched_flights['date'] = sched_flights.arrowdate.apply(lambda x: x.format('YYYY-MM-DD'))
bel_sched_flights = sched_flights[sched_flights["actual_ident"]==""]
bel_sched_flights = bel_sched_flights[["ident","aircrafttype", "origin", "destination",                            "seats_cabin_coach", "date", "departuretime"]]

citypairs = bel_sched_flights[["origin", "destination"]].drop_duplicates()

ac = bel_sched_flights.groupby(["aircrafttype"])
aircraft = ac.aggregate(np.mean)['seats_cabin_coach']

import numpy as np
from numpy import pi
def distance(lat1, lon1, lat2, lon2):
    # http://www.movable-type.co.uk/scripts/latlong.html
    R = 6371 / 1.6 # in status miles
    lat1 *= pi/180.
    lon1 *= pi/180.
    lat2 *= pi/180.
    lon2 *= pi/180.
    return R*np.arccos(
        np.sin(lat1)*np.sin(lat2) + 
        np.cos(lat1)*np.cos(lat2)*np.cos(lon2-lon1))

def airdistance(origin = "EBBR", destination = "EBBR"):
    if origin in airports and destination in airports:
        lat1 = airports[origin]["latitude"]
        lat2 = airports[destination]["latitude"]
        lon1 = airports[origin]["longitude"]
        lon2 = airports[destination]["longitude"]
        return int(distance(lat1, lon1, lat2, lon2))
    else:
        print("Not all airports known:", origin, "/",destination)
        return 0

def airdist(citypair):
    return airdistance(citypair["origin"], citypair["destination"])


def extract_coord(s):
    elem = s.split()
    t = (elem[1], elem[2]) # a tuple with two elements
    return t

def find_nearest(c):
    dist = 10000 # arbitrary high value
    airp = ""
    for a, aa in airports.items():
        d = distance(aa['latitude'], aa['longitude'], float(c[0]), float(c[1]))
        if d < dist:
            dist = d
            airp = a
    return airp

def validate_origin(o):
    if len(o)==4:    # rudimentary, but ICAO codes are 4 digits, good enough
        return o
    else:
        return find_nearest(extract_coord(o))

hist_flights['origin'] = hist_flights['origin'].apply(validate_origin)

def validate_city(city, icao):
    if not city:
        if icao in airports:
            city = airports[icao]['name'] # note that location is empty in the airports db, too
            print(city, icao)
    return city

hist_flights["originCity"] = hist_flights.apply(lambda r: validate_city(r['originCity'], 
                                                                       r['origin']), axis=1)
hist_flights["destinationCity"] = hist_flights.apply(lambda r: validate_city(r['destinationCity'], 
                                                                       r['destination']), axis=1)

origin = set(["EBBR", "EBLG", "EBAW", "EDDF", "LSZH"]) # special airports considered homebase
def decide_destination(flight):
    A = set([flight["origin"], flight["destination"]])
    B = set(["EBBR"]) # without the square brackets the set would contain E, B, R
    if A <= origin:   # is A wholly contained in list if 5?
        # then substract EBBR from it
        return A.difference(B).pop() # easy way to convert set of one element to element
    else: # at least one was not in the list, so remove the list to find it.
        # if both were in the list, select one arbitrarily
        return A.difference(origin).pop() # potentially one arbitrary of two elements
    
test_flight = {"origin": "LSZH", "destination" : "KIFR"}
decide_destination(test_flight)
    

hist_flights["udest"] = hist_flights.apply(decide_destination, axis=1)
bel_sched_flights["udest"] = bel_sched_flights.apply(decide_destination, axis=1)

hist_flights.loc[hist_flights['udest']=="GUCY"]

citypairs["dist"] = 0
citypairs["dist"] = citypairs.apply(airdist, axis=1)

def available_seat_miles(row):
    origin = row['origin']
    destination = row['destination']
    seats = row['seats_cabin_coach']
    dist = airdistance(origin, destination)
    if not dist:
        print("WARNING: no dist for ", origin, " ", destination)
    return  dist * seats

bel_sched_flights["a_s_m"] = bel_sched_flights.apply(available_seat_miles, axis=1)

bel_sched_flights[bel_sched_flights['udest']=="GUCY"]

t1 = arrow.get(2016, 3, 29).timestamp
t2 = arrow.get(2016, 4, 6).timestamp

BSF = bel_sched_flights.loc[(bel_sched_flights['departuretime'] > t1) & 
                      (bel_sched_flights['departuretime'] < t2)]
np.floor(BSF.a_s_m.sum() / 1e6)

g1 = BSF.groupby(['udest'])

def myroundsum(x):
    return np.round(np.sum(x)/1e6, decimals=1)

sched_agg = g1.aggregate(myroundsum)
sched_agg = sched_agg.combine_first(ap).sort_values(by="a_s_m", ascending=False)
sched_agg.ix[0:19, [0, 3, 5]]

hist_flights = hist_flights.join(aircraft, on="aircrafttype")

BHF = hist_flights.loc[(hist_flights['actualdeparturetime'] > t1) & 
                      (hist_flights['actualdeparturetime'] < t2), \
        ['ident','udest','ADT','origin','destination', 'originCity', 'destinationCity', 'seats_cabin_coach']]

def seat_miles_flown(row):
    origin = row['origin']
    destination = row['destination']
    seats = row['seats_cabin_coach']
    dist = airdistance(origin, destination)
    if not dist:
        print("WARNING: no dist for ", origin, " ", destination)
    return  dist * seats

BHF["smf"] = BHF.apply(seat_miles_flown, axis=1)
"Total seat miles flown in millions: {}".format(np.floor(BHF.smf.sum() / 1e6))

bhf_g = BHF.groupby(["ident", "udest", "ADT"])
bhf_c = bhf_g.count()['smf']
bhf_c[bhf_c>1]


hist_agg = BHF.groupby(['udest']).aggregate(np.sum)
hist_agg.smf = hist_agg.smf.apply(lambda x: np.round(x / 1e6, decimals=1))
hist_agg.combine_first(ap).sort_values(by="smf", ascending=False).ix[0:19, [1, 3, 5]]

import requests
def AirportInfo(icao):
    username = 'REMOVED'
    apiKey = "REMOVED"
    baseURL = "http://flightxml.flightaware.com/json/FlightXML2/"
    method = "AirportInfo"
    data = {"airportCode" : icao}
    r = requests.post(baseURL+method, auth=(username, apiKey), data=data)
    if r.status_code == 200:
        res = r.json()
        return res['AirportInfoResult']
    else:
        print("Problem with airport:", icao)

AirportInfo("EGCC")

combo = sched_agg.combine_first(hist_agg)[['a_s_m', 'smf']]
combo["ratio"] = combo['smf']/combo['a_s_m']

select = pd.notnull(combo["ratio"])
combo.combine_first(ap)[["name", "a_s_m", "smf", "ratio"]].sort_values(by="ratio", ascending=False)

def investigate(udest):
    print("--- SCHEDULED FLIGHTS ---")
    print(BSF.loc[BSF['udest'] == udest, ["date","ident", "origin", "destination"]])
    print("--- HISTORICAL FLIGHTS ---")
    print(BHF.sort_values(by="ADT").loc[BHF['udest'] == udest, ["ADT","ident", "origin", "destination"]])
    
investigate("GOOY")

ap = pd.DataFrame(airports).T

import sklearn.cluster
k = sklearn.cluster.KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, tol=0.0001, 
                           precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1)
df = ap[["latitude", "longitude"]]
df
ki = k.fit_predict(df)
ap["continent"] = ki

ap.loc[ap.continent==1]

ap.loc[ap.continent==0]

for kc in k.cluster_centers_:
    print(find_nearest(kc))

import sklearn.cluster
co = combo.fillna(0)
k = sklearn.cluster.KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, tol=0.0001, 
                           precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1)
ki = k.fit_predict(co)
ki

combo["category"] = ki
combo



