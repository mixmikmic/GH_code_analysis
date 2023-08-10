import requests
import json

BusRoutes = json.load(open('./bus_routes_from_data_mall.json'))

BusStops = json.load(open('./combined_bus_stops.json'))

def uniq_by(ls, key=lambda x: x):
    s = set()
    
    for item in ls:
        k = key(item)
        
        if k in s:
            continue
        else:
            s.add(k)
            yield item

def dedup_successive(ls):
    return [x for x,y in it.groupby(ls)]

import itertools as it

stopsByRoute = []

for (service, direction), route_stops in it.groupby(BusRoutes, lambda x : (x['ServiceNo'], x['Direction'])):
    sorted_route_stops = sorted(
        uniq_by(route_stops, key=lambda x:x['StopSequence']),
        key=lambda x:x['StopSequence']
    )
    
    stopsByRoute.append(((service, direction), sorted_route_stops))
    
filter(lambda x:x[0][0] == '410', stopsByRoute)

# for each route, emit (this stop, next stop)

# obtain the (prev, next) pairs
def prev_next(bus_stops):
    for a,b in zip(bus_stops, bus_stops[1:]):
        if a['BusStopCode'] == b['BusStopCode']:
            raise ValueError((a['BusStopCode'], bus_stops))
            
        yield a['BusStopCode'], b['BusStopCode']

stop_pairs = [
    (prev, next)
    for sbr in stopsByRoute
    for prev, next in prev_next(sbr[1])
]
# Exclude the non-stopping ones
stop_pairs = [x for x in stop_pairs if not x[0].startswith('E') and not x[1].startswith('E')]
stop_pairs = list(set(stop_pairs))

stop_pairs[1]

filter(lambda (x,y): x == y, stop_pairs)

import pyproj
svy = pyproj.Proj(init='epsg:3414')



def make_request(pair):
    ll1 = BusStops[pair[0]]
    ll2 = BusStops[pair[1]]
    
    return ['%s,%s' % (ll1['Latitude'], ll1['Longitude']),
            '%s,%s' % (ll2['Latitude'], ll2['Longitude'])]


def get_route(stop_pair):
    import polyline
    
    res = requests.get('http://localhost:8989/route',
        params={
            "point": [
                make_request(stop_pair)
            ],
            "type": "json",
            "vehicle": "car",
            "weighting": "fastest",
        })
    
    res = json.loads(res.text)
    
    if not 'paths' in res:
        return None
    
    return polyline.decode(res['paths'][0]['points'])

def has_stop_location(stop_pair):
    return stop_pair[0] in BusStops         and stop_pair[1] in BusStops

pair_routes = [(get_route(stop_pair), stop_pair) for stop_pair in stop_pairs if has_stop_location(stop_pair)]
pair_routes = [x for x in pair_routes if x[0] != None]

[a for a in pair_routes if a[1][1] == '14409'], [s for s in BusStops.values() if s['BusStopCode'] == '15049'], [s for s in BusStops.values() if s['BusStopCode'] == '14419'], 

# Display a routed path

import folium

m = folium.Map(location=[1.38, 103.8], zoom_start=12)
folium.PolyLine(pair_routes[100][0]).add_to(m)
m

import math

def swap((x,y)):
    return (y,x)

def heading(ll1, ll2):
    xy1 = svy(*swap(ll1))
    xy2 = svy(*swap(ll2))
    
    return (xy2[0] - xy1[0], xy2[1] - xy1[1])

deduped_pair_routes = [
    (dedup_successive(path), stops) for path, stops in pair_routes if len(dedup_successive(path)) >= 2
]

beginning_headings = [
    (heading(path[0], path[1]), stops[0]) for path, stops in deduped_pair_routes
]
end_headings = [
    (heading(path[-2], path[-1]), stops[1]) for path, stops in deduped_pair_routes
]

headings = beginning_headings + end_headings

stop_headings = {
    x: list([z[0] for z in y])
    for x,y in it.groupby(sorted(headings,
                                 key=lambda x:x[1]),
                          key=lambda x:x[1])
}

stop_headings['01029']

multiple = [(s,x) for s,x in stop_headings.items() if len(x) > 1]

def cosine_distance((a,b), (c,d)):
    return (a*c + b*d) / math.sqrt(a*a + b*b) / math.sqrt(c*c + d*d)

def cosine_distance_to_first(ls):
    
    return [
        cosine_distance(ls[0], l) for l in ls[1:]
    ]

multiple_headings = [ (s, cosine_distance_to_first(l)) for s,l in multiple]

import numpy as np

with_discrepencies = dict([
        (s, similarities)
        for s, similarities in multiple_headings
        if np.min(similarities) < 0.9
    ])
with_discrepencies

import numpy as np

# Get the final list...
def average_heading(xys):
    acc = np.array([0.0, 0.0])
    
    for xy in xys:
        xya = np.array(xy)
        xy_normalized = xy / np.linalg.norm(xya)
        
        acc += xy_normalized
        
    return math.atan2(*acc)

stop_average_headings = dict([ 
    (s, average_heading(x))
    for s, x in stop_headings.items()
    if s not in with_discrepencies
])
stop_average_headings

def add_heading(d, h):
    return dict(
        d.items() + [(u'Heading', None if h is None else h / math.pi * 180)]
    )

bus_stops_with_headings = [
    add_heading(s, stop_average_headings.get(s['BusStopCode']))
    for s in BusStops.values()
]
bus_stops_with_headings

json.dump(bus_stops_with_headings, open('bus_stops_with_headings.json', 'w'), indent=2)

def plot_stop_heading(bus_stop):
    c = BusStops[bus_stop]
    
    loc = (c['Latitude'], c['Longitude'])
    m = folium.Map(location=loc, zoom_start=18)

    # Draw the marker
    folium.Marker(location=loc).add_to(m)

    # Draw the line segment showing the direction
    def next_point(ll, h):
        xy = svy(*swap(ll))

        print xy

        x2 = xy[0] + 100 * math.sin(h)
        y2 = xy[1] + 100 * math.cos(h)

        return svy(x2, y2, inverse=True)

    folium.PolyLine([
            loc,
            swap(next_point(loc, stop_average_headings[bus_stop]))
        ]).add_to(m)

    return m

# check the sanity of the calculated stop headings
bus_stop = BusStops.keys()[12]
plot_stop_heading(bus_stop)



