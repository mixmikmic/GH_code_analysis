import requests
import pandas as pd
import numpy as np
from datetime import datetime

# defining constants for ease-of-use
lng_key = "MonitoredVehicleJourney_VehicleLocation_Longitude"
lat_key = "MonitoredVehicleJourney_VehicleLocation_Latitude"
bearing_key="MonitoredVehicleJourney_Bearing"
direction_key="MonitoredVehicleJourney_DirectionRef"
progress_key='MonitoredVehicleJourney_ProgressRate'
line_key=u'MonitoredVehicleJourney_LineRef'
dist_from_base_key = "MonitoredVehicleJourney_MonitoredCall_Extensions_Distances_CallDistanceAlongRoute"
dist_to_next_stop_key = u'MonitoredVehicleJourney_MonitoredCall_Extensions_Distances_DistanceFromCall'
timestamp_key = "RecordedAtTime"
join_key = "MonitoredVehicleJourney_FramedVehicleJourneyRef_DatedVehicleJourneyRef"

MTA_API_KEY="7a186bb8-a8f5-4fad-a8b7-6ec4dcd79505"
MTA_API_BASE = "http://bustime.mta.info/api/siri/vehicle-monitoring.json"
def _flatten_dict(root_key, nested_dict, flattened_dict):
    for key, value in nested_dict.iteritems():
        next_key = root_key + "_" + key if root_key != "" else key
        if isinstance(value, dict):
            _flatten_dict(next_key, value, flattened_dict)
        else:
            flattened_dict[next_key] = value
    return flattened_dict
    
#This is useful for the live MTA Data
params = {"key": MTA_API_KEY, }
BUS_ID="MTA NYCT_M116"
params["LineRef"] = BUS_ID
def nyc_current():
    resp = requests.get(MTA_API_BASE, params=params).json()
    info = resp['Siri']['ServiceDelivery']['VehicleMonitoringDelivery'][0]['VehicleActivity']
    return pd.DataFrame([_flatten_dict('', i, {}) for i in info])

import geopy
from geopy import Point
from geopy.distance import vincenty
from geopy.distance import VincentyDistance
from collections import namedtuple

KFTuple = namedtuple("KFTuple", ["model", "states", "means", "covariances", "update_times", "errors"])
KFTuple.__new__.__defaults__ = (None,) * len(KFTuple._fields)
time_step = 5 # in seconds

measurements_by_ref = {} # table of bus positions by route ID
def update_route_measurements(cur_data):
    route_refs = []
    for row in cur_data.iterrows():
        info = row[1]
        route_ref = info[join_key]
        route_refs.append(route_ref)
        if route_ref not in measurements_by_ref:
            measurements_by_ref[route_ref] = KFTuple(states=[info])
        if measurements_by_ref[route_ref].states[-1][timestamp_key] == info[timestamp_key]:
            continue # we're at same measurement
      
        measurements_by_ref[route_ref].states.append(info)
    
    return "OK"

def update_bus_info():
    cur_info =  nyc_current()
    cur_info[timestamp_key] = cur_info[timestamp_key].apply(iso_to_datetime)
    update_route_measurements(cur_info)
    for route, kf_tuple in measurements_by_ref.iteritems():
        measurements_by_ref[route] = update_model(kf_tuple)

def iso_to_datetime(ts_string):
    return datetime.strptime(ts_string.split(".")[0], "%Y-%m-%dT%H:%M:%S")

        
from pykalman import KalmanFilter
def init_model(kf_tuple):
    init = kf_tuple.states[-1]
    # TODO: seed average bus speed based on model
    v_lat, v_lng = v_to_components(5.5 / 3600., # assume bus is going 5.5mph to start [avg line speed]
                                   init[bearing_key],
                                   init[lat_key],
                                   init[lng_key])
    init['v_lat'] = v_lat
    init['v_lng'] = v_lng
    initial_state = kf_tuple.states[-1][[lat_key, 
                                        lng_key, 
                                        "v_lat",
                                        "v_lng"]]
    model = KalmanFilter(initial_state_mean=initial_state,
                         initial_state_covariance=np.eye(4),
                         transition_matrices=transition_matrix(5),
                         n_dim_obs=4)
    return KFTuple(model, kf_tuple.states, [initial_state], [np.eye(4)], [init[timestamp_key]], [])

def update_model(kf_tuple):
    model = kf_tuple.model 
    
    if model is None:
        return init_model(kf_tuple)
    
    latest = kf_tuple.states[-1]
    # nothing to do
    if latest[timestamp_key] == kf_tuple.update_times[-1]:
        mean, cov = kf_tuple.model.filter_update(kf_tuple.means[-1],
                                                 kf_tuple.covariances[-1])
        kf_tuple.means.append(mean)
        kf_tuple.covariances.append(cov)
        return kf_tuple
    else:
        cur = kf_tuple.states[-1]
        v_lat, v_lng = v_to_components(v_estimate(kf_tuple.states), # average bus speed, miles per second
                                   cur[bearing_key],
                                   cur[lat_key],
                                   cur[lng_key])
        cur['v_lat'] = v_lat
        cur['v_lng'] = v_lng
        cur_state = kf_tuple.states[-1][[lat_key, 
                                        lng_key, 
                                        "v_lat",
                                        "v_lng"]]
        mean, cov = kf_tuple.model.filter_update(kf_tuple.means[-1],
                                                 kf_tuple.covariances[-1],
                                                 cur_state)
        interp_mean, interp_cov = kf_tuple.model.filter_update(kf_tuple.means[-1],
                                                 kf_tuple.covariances[-1])
        
        kf_tuple.errors.append(forecast_error(mean, interp_mean))
        kf_tuple.means.append(mean)
        kf_tuple.covariances.append(cov)
        kf_tuple.update_times.append(latest[timestamp_key])
        return kf_tuple

    
import numpy as np
def transition_matrix(dt):
    T = np.eye(4)
    T[0][2] = dt
    T[1][3] = dt
    return T

from geopy.distance import vincenty
def v_estimate(states):
    # average bus speed
    if len(states) == 1:
        return 5.5 / 3600.
    else:
        p1 = (states[-2][lat_key], states[-2][lng_key])
        p2 = (states[-1][lat_key], states[-1][lng_key])
        t1 = states[-2][timestamp_key]
        t2 = states[-1][timestamp_key]
        return vincenty(p1, p2).miles / float((t2 - t1).seconds)

def forecast_error(mean, interp_mean):
    p1 = tuple(mean[:2])
    p2 = tuple(interp_mean[:2])
    return vincenty(p1, p2).miles

# convert v, bearing to v_x, v_y components
def v_to_components(v, bearing, lat, lng):
    d = VincentyDistance(miles=v) # num miles / second
    res = d.destination(geopy.Point(lat, lng), 90 - bearing) # nyc uses east / counterclockwise convention
    out = (res.latitude - lat, res.longitude - lng)
    return out


def run():
    import time
    for i in range(50):
        update_bus_info()
        time.sleep(time_step)
        print "iteration {}".format(i)

import dill
with open('measurements3.pkl', 'w+') as out_f:
    dill.dump(measurements_by_ref, out_f)

print "\n".join([str((k, len(v.states))) for k, v in measurements_by_ref.iteritems()])
route_id="MV_A5-Weekday-SDon-114500_M116_316"

import ujson as json
def kftuple_to_json(kf_tuple):
    output_dict = {"actual": [(x[timestamp_key], x[lat_key], x[lng_key], x[bearing_key]) for x in kf_tuple.states],
                   "preds": [(m, c) for m, c in zip(kf_tuple.means, zip(kf_tuple.covariances))]}
    return json.dumps(output_dict)

# kftuple_to_json(measurements_by_ref[route_id])

f = open("visualization/static/test.json", "w+")
f.write(kftuple_to_json(measurements_by_ref[route_id]))
f.close()

kf_tuple = measurements_by_ref[route_id]
[(x[lat_key], x[lng_key]) for x in kf_tuple.states]
# [(x[0], x[1], x[2], x[3]) for x in kf_tuple.means]

for i, t in measurements_by_ref.iteritems():
    print "========{}======".format(i)
    print "\n".join([str((x[lat_key], x[lng_key])) for x in t.states])

def set_live_keys():
    to_update = {
        "lng_key":"MonitoredVehicleJourney_VehicleLocation_Longitude",
        "lat_key": "MonitoredVehicleJourney_VehicleLocation_Latitude",
        "bearing_key":"MonitoredVehicleJourney_Bearing",
        "direction_key":"MonitoredVehicleJourney_DirectionRef",
        "progress_key":'MonitoredVehicleJourney_ProgressRate',
        "line_key":u'MonitoredVehicleJourney_LineRef',
        "dist_from_base_key": "MonitoredVehicleJourney_MonitoredCall_Extensions_Distances_CallDistanceAlongRoute",
        "dist_to_next_stop_key": u'MonitoredVehicleJourney_MonitoredCall_Extensions_Distances_DistanceFromCall',
        "timestamp_key": "RecordedAtTime",
        "join_key": "MonitoredVehicleJourney_FramedVehicleJourneyRef_DatedVehicleJourneyRef"
    }
    globals().update(to_update)
    
def set_back_keys():
    to_update = {
        "lat_key": "latitude",
        "lng_key": "longitude",
        "bearing_key": "bearing",
        "direction_key": "direction_id",
        "progress_key":  "progress",
        "line_key": "route_id",
        "dist_from_base_key": "dist_along_route",
        "dist_to_next_stop_key": "dist_from_stop",
        "timestamp_key": "timestamp",
        "join_key": "trip_id",
    }
    globals().update(to_update)

set_back_keys()

set_live_keys()

lat_key

from datetime import timedelta
def run_simulation(df):
    """
    :param df -- **the merged dataframe (on trips.txt)**
    """
    # this is idempotent
    set_back_keys()
    base_time = df[timestamp_key].min()
    prev_time = base_time
    for i in xrange(1000):
        if not (i % 10):
            print "{} seconds elapsed".format(i)
        delta = timedelta(seconds=i)
        current_time = base_time + delta
        cur_info = df[(df[timestamp_key] == current_time) & (df[line_key] == "M116")]
        if cur_info.empty:
            continue
        update_route_measurements(cur_info)
        for route, kf_tuple in measurements_by_ref.iteritems():
            measurements_by_ref[route] = update_model(kf_tuple)

    return "Done"

measurements_by_ref = {}
# merged = parse_historical_input("hello")
run_simulation(merged)

import pandas as pd
def parse_historical_input(filename):
    set_back_keys()
    df = pd.read_csv("bus_time_20150128.csv")
    trips = pd.read_csv("trips.txt")
    merged = df.merge(trips, on="trip_id")
    merged[timestamp_key] = pd.to_datetime(merged[timestamp_key], infer_datetime_format=True)
    return merged

vincenty((40.799888, -73.94462),
(40.799717, -73.944213)).miles

kt = measurements_by_ref.get(measurements_by_ref.keys()[0])

sum(kt.errors)



