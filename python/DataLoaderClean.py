get_ipython().magic('matplotlib inline')
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import json
from dateutil.parser import parse
import re
import matplotlib.pyplot as plt
import matplotlib
from xgboost import plot_importance
import requests
from sklearn.cross_validation import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
import time
import datetime
from datetime import date
from bayes_opt import BayesianOptimization
from sklearn.cross_validation import cross_val_score
import numpy as np
import pytz
from sklearn.tree import DecisionTreeClassifier
import xml.etree.ElementTree as ET
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import itertools
from sklearn.learning_curve import learning_curve
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import urllib.parse
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from os import path

# {"querytype":"occupancy","querytime":"2016-07-27T20:05:51+02:00",
# "post":{"connection":"http://irail.be/connections/8813003/20160727/IC1518",
# "from":"http://irail.be/stations/NMBS/008813003",
# "date":"Sun Jan 18 1970 01:14:03 GMT+0100 (CET)",
# "vehicle":"http://irail.be/vehicle/IC1518",
# "occupancy":"http://api.irail.be/terms/high"},
# "user_agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 
# (KHTML, like Gecko) Chrome/52.0.2743.82 Safari/537.36"}
_columns = ['querytime', 'seconds_since_midnight', 'hour', 'weekday', 'month', 'connection', 
            'from', 'from_string', 'from_lat', 'from_lng', 'morning_jam', 'evening_jam',
            'to', 'to_string', 'to_lat', 'to_lng', 'vehicle', 'vehicle_type', 'occupancy',
            'year', 'day', 'quarter']

stations_df = pd.read_csv('stations.csv')
stations_df = stations_df[['URI','name', 'latitude', 'longitude']]
stations_df['URI'] = stations_df['URI'].apply(lambda x: x.split('/')[-1])
week_day_mapping = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}

def parse_file(path):
    parsed_logs = []
    faulty_logs = 0
    time_zones = []
    with open(path) as data_file:  
        for line in data_file:
            occ_logline = json.loads(line)
            morning_commute = 0
            evening_commute = 0
            commute_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
            # Do a check if the querytype is occupancy
            if occ_logline['querytype'] == 'occupancy' and 'error' not in occ_logline             and 'querytime' in occ_logline:

                    try:
                        query_time = occ_logline['querytime']
                        try:
                            parsed_query_time = parse(query_time)
                            week_day = week_day_mapping[parsed_query_time.weekday()]
                            weekday_nr = parsed_query_time.weekday()
                            midnight = parsed_query_time.replace(hour=0, minute=0, second=0, microsecond=0)
                            seconds_since_midnight = (parsed_query_time - midnight).seconds
                            day = parsed_query_time.day
                            year = parsed_query_time.year
                            month = parsed_query_time.month
                            hour = parsed_query_time.hour
                            quarter = int(parsed_query_time.minute/15)
                            timezone_offset = parsed_query_time.tzinfo._offset
                            time_zones.append(timezone_offset)
                            hours_offset, remainder = divmod(timezone_offset.seconds, 3600)
                            # De ochtendspits valt doorgaans in de periode van 7.00 tot 9.00 uur. 
                            # De avondspits valt in de regel tussen 16.30 en 18.30 uur.
                            if 6 < (hour - hours_offset + 1) < 10 and week_day in commute_days:
                                morning_commute = 1
                            if 15 < (hour - hours_offset + 1) < 19 and week_day in commute_days:
                                evening_commute = 1
                        except ValueError:
                            faulty_logs += 1
                            continue

                        from_id = occ_logline['post']['from'].split('/')[-1]
                        to_id = occ_logline['post']['to'].split('/')[-1]
                        vehicle_id = occ_logline['post']['vehicle'].split('/')[-1]
                        occupancy = occ_logline['post']['occupancy'].split('/')[-1]
                        connection = occ_logline['post']['connection']
                        from_string = stations_df[stations_df['URI'] == from_id]['name'].values[0]
                        to_string = stations_df[stations_df['URI'] == to_id]['name'].values[0]
                        from_lat = stations_df[stations_df['URI'] == from_id]['latitude'].values[0]
                        from_lng = stations_df[stations_df['URI'] == from_id]['longitude'].values[0]
                        to_lat = stations_df[stations_df['URI'] == to_id]['latitude'].values[0]
                        to_lng = stations_df[stations_df['URI'] == to_id]['longitude'].values[0]
                        
                        if from_id[:2] == '00' and to_id[:2] == '00' and vehicle_id != 'undefined'                         and len(to_id) > 2 and len(from_id) > 2:
                            pattern = re.compile("^([A-Z]+)[0-9]+$")
                            vehicle_type = pattern.match(vehicle_id).group(1)
                            parsed_logs.append([parsed_query_time, seconds_since_midnight, hour, week_day, month,
                                                connection, from_id, from_string, from_lat, from_lng, morning_commute,
                                                evening_commute, to_id, to_string, to_lat, to_lng, vehicle_id, 
                                                vehicle_type, occupancy, year, day, quarter])
                        else:
                            faulty_logs += 1
                    except Exception as e:
                        faulty_logs += 1
                        continue
        return parsed_logs, faulty_logs
                    
parsed_file1, faulty1 = parse_file('occupancy-2016-10-10.newlinedelimitedjsonobjects.jsonstream')
parsed_file2, faulty2  = parse_file('occupancy-until-20161029.newlinedelimitedjsonobjects')
parsed_file3, faulty3  = parse_file('occupancy-until-20161219.nldjson')
logs_df = pd.DataFrame(parsed_file1+parsed_file2+parsed_file3)
logs_df.columns = _columns
old_length = len(logs_df)
print(faulty1+faulty2+faulty3, 'logs discarded ---', old_length, 'parsed')
logs_df = logs_df.drop_duplicates(subset=['querytime', 'from', 'to', 'vehicle'])
print(old_length - len(logs_df), 'real duplicates removed')

old_length = len(logs_df)
logs_df = logs_df.reset_index(drop=True)
logs_df['index'] = logs_df.index
filtered_df = logs_df[['vehicle', 'from', 'day', 'month', 'hour', 'occupancy', 'index']].groupby(by=['vehicle', 'from', 'day', 'month', 'hour'], as_index=False)[['vehicle', 'from', 'day', 'month', 'hour', 'occupancy', 'index']].agg({'occupancy': lambda x:x.value_counts().index[0]})
logs_df = logs_df.loc[filtered_df['occupancy']['index'].values]
#print(filtered_df['occupancy'][['index', 'occupancy']])
#print(logs_df['occupancy'])
print(old_length - len(logs_df), 'harder duplicates removed')

features_df = logs_df[['seconds_since_midnight', 'weekday', 'from_string', 'to_string', 'vehicle_type', 
                       'month', 'from_lat', 'from_lng', 'to_lat', 'to_lng']]
features_df = pd.get_dummies(features_df, columns=['weekday', 'from_string', 'to_string', 'vehicle_type'])
print('Features dataframe dimensions:', len(features_df), 'x', len(features_df.columns))
occupancy_mapping = {'low': 0, 'medium': 1, 'high': 2}
labels_df = logs_df['occupancy'].map(occupancy_mapping) 

xgb = XGBClassifier(learning_rate=0.15, n_estimators=500,
                     gamma=0.25, subsample=0.75, colsample_bytree=0.7,
                     nthread=1, reg_lambda=0.25,
                     min_child_weight=5, max_depth=9, objective='multi:softprob')

NR_FOLDS = 5
NR_FEATURES = 40

skf = StratifiedKFold(labels_df.values, n_folds=NR_FOLDS, shuffle=True, random_state=1337)
accuracies = []
accuracies_no_drukte = []
for fold, (train_idx, test_idx) in enumerate(skf):
    print ('Fold', fold+1, '/', NR_FOLDS)
    X_train = features_df.iloc[train_idx, :].reset_index(drop=True)
    y_train = labels_df.iloc[train_idx].reset_index(drop=True)
    X_test = features_df.iloc[test_idx, :].reset_index(drop=True)
    y_test = labels_df.iloc[test_idx].reset_index(drop=True)
    train = X_train.copy()
    train[y_train.name] = y_train
    
    
    #xgb = construct_classifier(train, X_train.columns, y_train.name)
    xgb.fit(X_train, y_train)
    selected_features_idx = xgb.feature_importances_.argsort()[-NR_FEATURES:][::-1]
    plt.bar(range(len(selected_features_idx)), [xgb.feature_importances_[i] for i in selected_features_idx])
    plt.xticks(range(len(selected_features_idx)), [features_df.columns[i] for i in selected_features_idx], rotation='vertical')
    plt.show()

    predictions = xgb.predict(X_test)
    conf_matrix = confusion_matrix(y_test, predictions)
    print(conf_matrix)
    accuracy = sum([conf_matrix[i][i] for i in range(len(conf_matrix))])/np.sum(conf_matrix)
    print('accuracy:', accuracy)
    accuracies.append(accuracy)
    
    
print('Avg accuracy:', np.mean(accuracies))
print('Avg accuracy no drukte:', np.mean(accuracies_no_drukte))

# Load all connections data
import glob
files = glob.glob('connections/data0810/*.json')
"""
            "departureStop": "http://irail.be/stations/NMBS/008821071", 
            "departureTime": "2016-07-31T22:23:00.000Z", 
            "departureDelay": 1260, 
            "gtfs:trip": "http://irail.be/trips/L257314091", 
            "arrivalStop": "http://irail.be/stations/NMBS/008821543", 
            "gtfs:route": "http://irail.be/vehicle/L2573", 
            "arrivalDelay": 1260, 
            "arrivalTime": "2016-07-31T22:23:00.000Z", 
            "@id": "http://irail.be/connections/8821071/20160801/L2573", 
            "@type": "Connection"
"""
connection_vectors = []
_columns = ['trip_id', 'vehicle', 'from_station', 'departure_time', 'departure_delay',
            'to_station', 'arrival_time', 'arrival_delay', 'connection_id', 'year', 'month', 'day']
for i,file in enumerate(files):
    print(i,'/',len(files))
    with open(file) as json_data:
        connection_json = json.load(json_data)
        for stop in connection_json["@graph"]:
            trip_id = stop['gtfs:trip']
            vehicle_name = stop['gtfs:route'].split('/')[-1]
            from_station_id = stop['departureStop'].split('/')[-1]
            departure_time = parse(stop['departureTime'])
            year = departure_time.year
            month = departure_time.month
            day = departure_time.day
            if 'departureDelay' in stop:
                departure_delay = stop['departureDelay']
            else:
                departure_delay = 0
            to_station_id = stop['arrivalStop'].split('/')[-1]
            arrival_time = parse(stop['arrivalTime'])
            if 'arrivalDelay' in stop:
                arrival_delay = stop['arrivalDelay']
            else:
                arrival_delay = 0
            connection_id = stop['@id']
            
            connection_vectors.append([trip_id, vehicle_name, from_station_id, departure_time,
                                       departure_delay, to_station_id, arrival_time, arrival_delay,
                                       connection_id, year, month, day])
connections_df = pd.DataFrame(connection_vectors)
connections_df.columns = _columns

URL_ROOT = 'http://graph.spitsgids.be/connections/?departureTime='
START = datetime.datetime(2016, 12, 1, 1, 0, 0)
END = datetime.datetime(2016, 12, 19, 0, 0, 0)
def retrieve_schedule(url_root, start, end, folder):
    current = start
    while current < end:
        ts = current.strftime('%Y-%m-%dT%H:%M')
        url = url_root + urllib.parse.quote(ts)
        response = requests.get(url)
        if response.ok:
            data = json.loads(str(response.content)[2:-1])
            filename = path.join(folder, ts + '.json')
            with open(filename, 'w') as outfile:
                json.dump(data, outfile, indent=4)
            print('%s stored' % url)
        else:
            print('%s failed' % url)

        current = current + datetime.timedelta(minutes=10)
retrieve_schedule(URL_ROOT, start=START, end=END, folder='connections/data0810')

# For each entry in the occupancy_df, filter out entries of the connections_df with:
# same vehicle number, same departure and same day/month
stations = np.unique(stations_df['URI'].values)
distance_feature_vectors = []
for i in range(len(logs_df)):
    print(str(i+1), '/', str(len(logs_df)))
    entry = logs_df.iloc[i,:]
    entry_station = entry['from']
    entry_time = entry['querytime']
    distances = {}
    for station in stations: distances[station] = None
    connections_entry = connections_df[(connections_df.from_station == entry['from'])
                                      & (connections_df.vehicle == entry['vehicle'])
                                      & (connections_df.year == entry['year'])
                                      & (connections_df.month == entry['month'])
                                      & (connections_df.day == entry['day'])]
    trip_id = None
    if len(connections_entry) > 0: 
        connections_entry = connections_entry.iloc[0,:]
        trip_id = connections_entry['trip_id']
    else: 
        print('Fault occured')
        
    if trip_id is not None:
        year_str = str(entry['year'])
        month_str = '0'+str(entry['month']) if entry['month'] < 10 else str(entry['month'])
        day_str = '0'+str(entry['day']) if entry['day'] < 10 else str(entry['day'])
        filtered_connections_stations = list(connections_df[(connections_df.connection_id.str.contains(year_str+
                                                                                                       month_str+
                                                                                                       day_str+
                                                                                                       '/'+entry['vehicle']))
                                                           & (connections_df.year == entry['year'])
                                                           & (connections_df.month == entry['month'])
                                                           & (connections_df.day == entry['day'])].sort_values(by='departure_time')['from_station'].values)
        print(year_str+month_str+day_str+'/'+entry['vehicle'],
              len(filtered_connections_stations))
        
        #if trip_id.split('/')[-1] != '':
        if entry_station in filtered_connections_stations:
            for station in filtered_connections_stations:
                #print(filtered_connections_stations.index(station) - filtered_connections_stations.index(entry_station))
                distances[station] = filtered_connections_stations.index(station) - filtered_connections_stations.index(entry_station)

            distances['time'] = entry_time
            distances['from_station'] = entry_station
            distance_feature_vectors.append({**distances, **entry.to_dict()})
        #else:
        #    print('wrong trip id')
        
distance_df = pd.DataFrame(distance_feature_vectors)

# --> Take the trip_id and filter out same trip_id on same day

# Create distance matrix and from this, create features

#print(sorted(map(lambda x: (pd.Timestamp(x).day, pd.Timestamp(x).month,
#                            pd.Timestamp(x).hour, pd.Timestamp(x).minute), 
#                 connections_df[connections_df.vehicle == 'IC826']['arrival_time'].values)))
print(len(distance_df))
distance_df.to_csv('distance_df.csv')

print(distance_df.head(5))
frequencies = {}
stations = np.unique(stations_df['URI'].values)
for station in stations:
    frequencies[station] = len(distance_df[distance_df[station] >= 0])

print(max(frequencies.values()))
stations = np.unique(stations_df['URI'].values)
new_columns = list(set(distance_df.columns) - set(stations))
distance_feature_vectors = []
for i in range(len(distance_df)):
    entry = distance_df.iloc[i,:]
    absolute_freq, weighted_freq, am_weighted_freq = 0, 0, 0
    for station in stations:
        if entry[station] is not None and entry[station] >= 0:
            absolute_freq += frequencies[station]
            w_freq = (frequencies[station] / max(abs(entry[station]), 1)) * np.sign(entry[station]) 
            weighted_freq += w_freq 
            if entry['hour'] < 12: am_weighted_freq += w_freq
            else: am_weighted_freq -= w_freq
                
    print(absolute_freq, weighted_freq, am_weighted_freq)
    distance_feature_vector = []
    for column in new_columns: 
        distance_feature_vector.append(entry[column])
    distance_feature_vector.append(absolute_freq)
    distance_feature_vector.append(weighted_freq)
    distance_feature_vector.append(am_weighted_freq)
    distance_feature_vectors.append(distance_feature_vector)
distance_feature_df = pd.DataFrame(distance_feature_vectors)
distance_feature_df.columns = new_columns + ['absolute_freq', 'weighted_freq', 'am_weighted_freq']
distance_feature_df.to_csv('distance_feature_df.csv')

# Maybe we should use only from_station, from_lat and from_lng? the to_ features seem somewhat unimportant? 
# We want to predict if the train will be full when the user gets on
features_df = distance_feature_df[['seconds_since_midnight', 'weekday', 'from_string', 'to_string', 'vehicle_type', 
                                   'month', 'from_lat', 'from_lng', 'to_lat', 'to_lng',
                                   'absolute_freq', 'weighted_freq', 'am_weighted_freq', 'evening_jam',
                                   'morning_jam']]
features_df = pd.get_dummies(features_df, columns=['weekday','from_string', 'to_string', 'vehicle_type'])
print('Features dataframe dimensions:', len(features_df), 'x', len(features_df.columns))
occupancy_mapping = {'low': 0, 'medium': 1, 'high': 2}
labels_df = distance_feature_df['occupancy'].map(occupancy_mapping)

xgb = XGBClassifier(learning_rate=0.075, n_estimators=1750,
                     gamma=0.9, subsample=0.75, colsample_bytree=0.7,
                     nthread=1, scale_pos_weight=1, reg_lambda=0.25,
                     min_child_weight=5, max_depth=13)

NR_FOLDS = 5
NR_FEATURES = 40

skf = StratifiedKFold(labels_df.values, n_folds=NR_FOLDS, shuffle=True, random_state=None)

for fold, (train_idx, test_idx) in enumerate(skf):
    print ('Fold', fold+1, '/', NR_FOLDS)
    X_train = features_df.iloc[train_idx, :].reset_index(drop=True)
    y_train = labels_df.iloc[train_idx].reset_index(drop=True)
    X_test = features_df.iloc[test_idx, :].reset_index(drop=True)
    y_test = labels_df.iloc[test_idx].reset_index(drop=True)
    
    xgb.fit(X_train, y_train)
    selected_features_idx = xgb.feature_importances_.argsort()[-NR_FEATURES:][::-1]
    plt.bar(range(len(selected_features_idx)), [xgb.feature_importances_[i] for i in selected_features_idx])
    plt.xticks(range(len(selected_features_idx)), [features_df.columns[i] for i in selected_features_idx], rotation='vertical')
    plt.show()

    predictions = xgb.predict(X_test)
    conf_matrix = confusion_matrix(y_test, predictions)
    print(conf_matrix)
    print('accuracy:', sum([conf_matrix[i][i] for i in range(len(conf_matrix))])/np.sum(conf_matrix))


distance_feature_df = pd.read_csv('distance_feature_df.csv')
features_df = distance_feature_df[['seconds_since_midnight', 'weekday', 'from_string', 'to_string', 'vehicle_type', 
                                   'month', 'from_lat', 'from_lng', 'to_lat', 'to_lng',
                                   'absolute_freq', 'weighted_freq', 'am_weighted_freq', 'evening_jam',
                                   'morning_jam']]
features_df = pd.get_dummies(features_df, columns=['weekday','from_string', 'to_string', 'vehicle_type'])
print('Features dataframe dimensions:', len(features_df), 'x', len(features_df.columns))
occupancy_mapping = {'low': 0, 'medium': 1, 'high': 2}
labels_df = distance_feature_df['occupancy'].map(occupancy_mapping)

xgb = XGBClassifier(learning_rate=0.075, n_estimators=1750,
                     gamma=0.9, subsample=0.75, colsample_bytree=0.7,
                     nthread=1, scale_pos_weight=1, reg_lambda=0.25,
                     min_child_weight=5, max_depth=13)

NR_FOLDS = 5
NR_FEATURES = 40

skf = StratifiedKFold(labels_df.values, n_folds=NR_FOLDS, shuffle=True, random_state=None)

for fold, (train_idx, test_idx) in enumerate(skf):
    print ('Fold', fold+1, '/', NR_FOLDS)
    X_train = features_df.iloc[train_idx, :].reset_index(drop=True)
    y_train = labels_df.iloc[train_idx].reset_index(drop=True)
    X_test = features_df.iloc[test_idx, :].reset_index(drop=True)
    y_test = labels_df.iloc[test_idx].reset_index(drop=True)
    
    params = {'n_neighbors':[3,5,10], 'algorithm': ['auto', 'ball_tree', 'kd_tree']}
    knn = GridSearchCV(KNeighborsClassifier() ,params, refit='True', n_jobs=1, cv=5)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    conf_matrix = confusion_matrix(y_test, predictions)
    print(conf_matrix)
    print('accuracy KNN:', sum([conf_matrix[i][i] for i in range(len(conf_matrix))])/np.sum(conf_matrix))
    
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    predictions = lda.predict(X_test)
    conf_matrix = confusion_matrix(y_test, predictions)
    print(conf_matrix)
    print('accuracy LDA:', sum([conf_matrix[i][i] for i in range(len(conf_matrix))])/np.sum(conf_matrix))
    
    #params = {'C':[1,5,0.1,0.01],'gamma':[0.05,0.5,1], 'kernel': ['rbf', 'linear', 'poly']}
    #svc = GridSearchCV(SVC() ,params, refit='True', n_jobs=1, cv=5)
    #svc.fit(X_train, y_train)
    #predictions = svc.predict(X_test)
    #conf_matrix = confusion_matrix(y_test, predictions)
    #print(conf_matrix)
    #print('accuracy SVM:', sum([conf_matrix[i][i] for i in range(len(conf_matrix))])/np.sum(conf_matrix))
    

distance_feature_df = pd.read_csv('distance_feature_df.csv')
print(len(distance_feature_df))
weather_delay_events_feature_df = pd.read_csv('weather_delay_features.csv')
feature_df = pd.merge(distance_feature_df, weather_delay_events_feature_df, how='left', 
                      on=['querytime', 'vehicle'])
feature_df.dropna(subset=['temperature_to'], inplace=True)
feature_df.drop_duplicates(subset=['time', 'vehicle'], inplace=True)
print(len(feature_df))
print(list(feature_df.columns))
__columns = ['querytime', 'seconds_since_midnight_x', 'morning_jam', 'from_lng_x', 'day', 'to_lng_x',
            'connection', 'evening_jam', 'vehicle_type_x', 'occupancy_x', 'to_lat_x', 'year', 
            'weekday_x', 'vehicle', 'from_lat_x', 'from_string_x', 'month_x', 'hour', 'to_string_x', 
            'absolute_freq', 'weighted_freq', 'am_weighted_freq', 'temperature_from', 'humidity_from', 
            'windspeed_from', 'visibility_from', 'weather_type_from', 'temperature_to', 'humidity_to', 
            'windspeed_to', 'visibility_to', 'weather_type_to', 'delay_15', 'delay_30', 'delay_60', 
            'delay_100']
for col in __columns:
    if col not in list(feature_df.columns): print(col)
feature_df = feature_df[__columns]
feature_df.to_csv('distance_weather_delay.csv')

# Maybe we should use only from_station, from_lat and from_lng? the to_ features seem somewhat unimportant? 
# We want to predict if the train will be full when the user gets on
features_df = pd.read_csv('distance_weather_delay.csv')
features_df = features_df.drop(['connection', 'vehicle', 'querytime', 'Unnamed: 0'], axis=1)
features_df = pd.get_dummies(features_df, columns=['weekday_x','from_string_x', 'to_string_x', 'vehicle_type_x',
                                                   'weather_type_to', 'weather_type_from'])
print('Features dataframe dimensions:', len(features_df), 'x', len(features_df.columns))
occupancy_mapping = {'low': 0, 'medium': 1, 'high': 2}
labels_df = features_df['occupancy_x'].map(occupancy_mapping)
features_df = features_df.drop(['occupancy_x'], axis=1)

from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel

xgb = XGBClassifier(learning_rate=0.075, n_estimators=1750,
                     gamma=0.9, subsample=0.75, colsample_bytree=0.7,
                     nthread=1, scale_pos_weight=1, reg_lambda=0.25,
                     min_child_weight=5, max_depth=13)

NR_FOLDS = 5
NR_FEATURES = 40

skf = StratifiedKFold(labels_df.values, n_folds=NR_FOLDS, shuffle=True, random_state=None)

accuracies = []
accuracies_fs = []
for fold, (train_idx, test_idx) in enumerate(skf):
    print ('Fold', fold+1, '/', NR_FOLDS)
    X_train = features_df.iloc[train_idx, :].reset_index(drop=True)
    y_train = labels_df.iloc[train_idx].reset_index(drop=True)
    X_test = features_df.iloc[test_idx, :].reset_index(drop=True)
    y_test = labels_df.iloc[test_idx].reset_index(drop=True)

    lsvc = LinearSVC(C=0.05, penalty="l1", dual=False).fit(X_train, y_train)
    model = SelectFromModel(lsvc, prefit=True)
    X_train_fs = model.transform(X_train)
    X_test_fs = model.transform(X_test)
    print(X_new.shape)
    
    xgb.fit(X_train, y_train)
    selected_features_idx = xgb.feature_importances_.argsort()[-NR_FEATURES:][::-1]
    plt.bar(range(len(selected_features_idx)), [xgb.feature_importances_[i] for i in selected_features_idx])
    plt.xticks(range(len(selected_features_idx)), [features_df.columns[i] for i in selected_features_idx], rotation='vertical')
    plt.show()

    predictions = xgb.predict(X_test)
    conf_matrix = confusion_matrix(y_test, predictions)
    print(conf_matrix)
    acc = sum([conf_matrix[i][i] for i in range(len(conf_matrix))])/np.sum(conf_matrix)
    print('accuracy:', acc)
    accuracies.append(acc)
    
    xgb.fit(X_train_fs, y_train)
    selected_features_idx = xgb.feature_importances_.argsort()[-NR_FEATURES:][::-1]
    plt.bar(range(len(selected_features_idx)), [xgb.feature_importances_[i] for i in selected_features_idx])
    plt.xticks(range(len(selected_features_idx)), [features_df.columns[i] for i in selected_features_idx], rotation='vertical')
    plt.show()

    predictions = xgb.predict(X_test_fs)
    conf_matrix = confusion_matrix(y_test, predictions)
    print(conf_matrix)
    acc = sum([conf_matrix[i][i] for i in range(len(conf_matrix))])/np.sum(conf_matrix)
    print('accuracy FS:', acc)
    accuracies_fs.append(acc)
    
print('Mean accuracy =', np.mean(accuracies))
print('Mean accuracy FS =', np.mean(accuracies_fs))



