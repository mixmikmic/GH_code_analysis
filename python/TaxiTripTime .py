import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
get_ipython().magic('matplotlib inline')
from datetime import datetime
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.linear_model import LinearRegression, Ridge,BayesianRidge
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import mean_squared_error
from math import radians, cos, sin, asin, sqrt

train = pd.read_csv("datasets/train.csv")
test = pd.read_csv("datasets/test.csv")

train.head()

test.head()

pd.set_option('display.float_format', lambda x: '%.3f' % x)
train.describe()

train.isnull().sum()

test.isnull().sum()

train['vendor_id'].hist()

train['vendor_id'].value_counts()

# vendorid having 2 seems to have a larger duration
fig,axes = plt.subplots(figsize = (5,5))
sns.barplot(data=train, x='vendor_id',y='trip_duration',hue='vendor_id')

# Number of people fom 1 to 6 almost have same time to travel as opposed to 7 to 9.
fig,axes = plt.subplots(figsize = (5,5))
sns.barplot(data=train, x='passenger_count',y='trip_duration',hue='passenger_count')

train = train[train['pickup_longitude'] <= -73.75]
train = train[train['pickup_longitude'] >= -74.03]
train = train[train['pickup_latitude'] <= 40.85]
train = train[train['pickup_latitude'] >= 40.63]
train = train[train['dropoff_longitude'] <= -73.75]
train = train[train['dropoff_longitude'] >= -74.03]
train = train[train['dropoff_latitude'] <= 40.85]
train = train[train['dropoff_latitude'] >= 40.63]
train.head()

train.shape

m = np.mean(train['trip_duration'])
s = np.std(train['trip_duration'])
print("Mean: {:.3f}  Standard Dev:{:.3f}".format(m,s))
Train = train[train['trip_duration'] <= m + 2*s]
Train = train[train['trip_duration'] >= m - 2*s]

Train.shape

sns.distplot(a=train['trip_duration'],bins=100)
plt.xlabel('trip_duration')
plt.ylabel('number of train records')
plt.show()

fig,axes = plt.subplots(figsize=(8,7))
plt.ylabel('KDE of train datasets')
plt.xlabel('log of Trip Duration')
sns.distplot(a=np.log1p(Train['trip_duration']),bins=100, fit=stats.norm)

Train['log_trip_duration'] = np.log1p(Train['trip_duration'])

Test = test

Train['pickup_datetime_converted'] =pd.to_datetime(Train['pickup_datetime']) 
Test['pickup_datetime_converted'] =pd.to_datetime(Test['pickup_datetime']) 

Train.loc[:, 'pickup_date'] = Train['pickup_datetime_converted'].dt.date
Test.loc[:, 'pickup_date'] = Test['pickup_datetime_converted'].dt.date

fig,axes = plt.subplots(figsize =(16,8))
plt.plot(Train.groupby('pickup_date').count()[['id']], 'x-', label='train')
plt.plot(Test.groupby('pickup_date').count()[['id']], 'x-', label='test')
plt.title('Pickups over given time frame.')
plt.legend(loc=0)
plt.ylabel('Number of pickups')

Train.head()

Train.groupby('vendor_id')['trip_duration'].mean()

Train.groupby('store_and_fwd_flag')['trip_duration'].mean()

Train.groupby('passenger_count')['trip_duration'].mean()

Train.groupby('passenger_count').size()

sns.lmplot(x='pickup_latitude',y = 'pickup_latitude',fit_reg=False,data=Train)

N = 100000
city_long_border = (-74.03, -73.75)
city_lat_border = (40.63, 40.85)
fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
ax[0].scatter(Train['pickup_longitude'].values[:N], Train['pickup_latitude'].values[:N],
              color='blue', s=1, label='train', alpha=0.1)
ax[1].scatter(Test['pickup_longitude'].values[:N], Test['pickup_latitude'].values[:N],
              color='green', s=1,label='test', alpha=0.1)
fig.suptitle('Train and test.')
ax[0].legend(loc=0)
ax[0].set_ylabel('latitude')
ax[0].set_xlabel('longitude')
ax[1].set_xlabel('longitude')
ax[1].legend(loc=0)
plt.ylim(city_lat_border)
plt.xlim(city_long_border)
plt.show()

def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b

def bearing_array(lat1, lng1, lat2, lng2):
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))

Train.loc[:, 'distance_haversine'] = haversine_array(Train['pickup_latitude'].values, Train['pickup_longitude'].values, Train['dropoff_latitude'].values, Train['dropoff_longitude'].values)
Test.loc[:, 'distance_haversine'] = haversine_array(Test['pickup_latitude'].values, Test['pickup_longitude'].values, Test['dropoff_latitude'].values, Test['dropoff_longitude'].values)    
    
Train.loc[:, 'distance_dummy_manhattan'] =  dummy_manhattan_distance(Train['pickup_latitude'].values, Train['pickup_longitude'].values, Train['dropoff_latitude'].values, Train['dropoff_longitude'].values)
Test.loc[:, 'distance_dummy_manhattan'] =  dummy_manhattan_distance(Test['pickup_latitude'].values, Test['pickup_longitude'].values, Test['dropoff_latitude'].values, Test['dropoff_longitude'].values)

Train.loc[:, 'direction'] = bearing_array(Train['pickup_latitude'].values, Train['pickup_longitude'].values, Train['dropoff_latitude'].values, Train['dropoff_longitude'].values)
Test.loc[:, 'direction'] = bearing_array(Test['pickup_latitude'].values, Test['pickup_longitude'].values, Test['dropoff_latitude'].values, Test['dropoff_longitude'].values)

Train

Test

fig,axes = plt.subplots()
axes.scatter(x=Train['distance_haversine'].values,y=Train['trip_duration'].values)
plt.xlabel("Haversine Distance")
plt.ylabel("Duration")

coords = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,
                    train[['dropoff_latitude', 'dropoff_longitude']].values))

coords

sample_ind = np.random.permutation(len(coords))[:500000]
kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords[sample_ind])

Train.loc[:, 'pickup_cluster'] = kmeans.predict(Train[['pickup_latitude', 'pickup_longitude']])
Train.loc[:, 'dropoff_cluster'] = kmeans.predict(Train[['dropoff_latitude', 'dropoff_longitude']])
test.loc[:, 'pickup_cluster'] = kmeans.predict(test[['pickup_latitude', 'pickup_longitude']])
test.loc[:, 'dropoff_cluster'] = kmeans.predict(test[['dropoff_latitude', 'dropoff_longitude']])

Train.head()

fig, ax = plt.subplots(ncols=1, nrows=1)
ax.scatter(Train.pickup_longitude.values[:500000], Train.pickup_latitude.values[:500000], s=10, lw=0,
           c=Train.pickup_cluster[:500000].values, cmap='autumn', alpha=0.2)
ax.set_xlim(city_long_border)
ax.set_ylim(city_lat_border)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.show()

Train['Month'] = Train['pickup_datetime_converted'].dt.month
test['Month'] = test['pickup_datetime_converted'].dt.month

Train.groupby('Month').count()['id'],test.groupby('Month').count()['id']

fig,axes = plt.subplots()
axes.scatter(x=Train['distance_dummy_manhattan'].values,y=Train['trip_duration'].values)
plt.xlabel("Manhattan Distance")
plt.ylabel("Trip Duration")

Train['Day'] = Train['pickup_datetime_converted'].dt.day
test['Day'] = test['pickup_datetime_converted'].dt.day

Train.head()

Train['Hour'] = Train['pickup_datetime_converted'].dt.hour
test['Hour'] = test['pickup_datetime_converted'].dt.hour

Train.head()

Train.groupby('Day').size() , test.groupby('Day').size()

Train['Hour'].value_counts().sort_index(ascending = True),test['Hour'].value_counts().sort_index(ascending = True)

plt.hist(Train['Hour'])

fig, axes = plt.subplots(ncols=2,nrows=1,sharey=True,figsize=(14,8))
axes[0].scatter(Train['Hour'].value_counts().index,Train['Hour'].value_counts().values)
axes[1].plot(Train['Hour'].value_counts().index,Train['Hour'].value_counts().values)

Train['h_average_speed'] = 1000 * Train['distance_haversine']/Train['trip_duration']
Train['m_avg_speed'] = 1000 * Train['distance_dummy_manhattan'] / train['trip_duration']

Train['m_avg_speed']

fig, ax = plt.subplots(ncols=3, sharey=True,figsize=(14,6))
ax[0].plot(Train.groupby('Hour').mean()['h_average_speed'], 'bo-', lw=2, alpha=0.7)
ax[1].plot(Train.groupby('Day').mean()['h_average_speed'], 'go-', lw=2, alpha=0.7)
ax[2].plot(Train.groupby('Month').mean()['h_average_speed'], 'ro-', lw=2, alpha=0.7)
ax[0].set_xlabel('Hour')
ax[1].set_xlabel('weekday')
ax[2].set_xlabel('Month')
ax[0].set_ylabel('average speed')
fig.suptitle('Rush hour average traffic speed')
plt.show()

plt.plot(Train.groupby('Day').mean()['h_average_speed'])

fig,axes = plt.subplots(figsize=(12,8))
plt.plot(Train.groupby('pickup_cluster').mean()['h_average_speed'])

fig,axes = plt.subplots(figsize=(12,8))
plt.plot(Train.groupby('dropoff_cluster').mean()['h_average_speed'])

Train['pickup_lat_bin'] = np.round(train['pickup_latitude'], 3)
Train['pickup_long_bin'] = np.round(train['pickup_longitude'], 3)

fr1 = pd.read_csv('datasets/fastest_routes_train_part_1.csv', usecols=['id', 'total_distance', 'total_travel_time',  'number_of_steps'])
fr2 = pd.read_csv('datasets/fastest_routes_train_part_2.csv', usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])
test_street_info = pd.read_csv('datasets/fastest_routes_test.csv',
                               usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])

train_street_info = pd.concat((fr1, fr2))
train = Train.merge(train_street_info, how='left', on='id')
test = test.merge(test_street_info, how='left', on='id')

train.shape, test.shape

vendor_train = pd.get_dummies(train['vendor_id'], prefix='vi', prefix_sep='_')
vendor_test = pd.get_dummies(test['vendor_id'], prefix='vi', prefix_sep='_')
passenger_count_train = pd.get_dummies(train['passenger_count'], prefix='pc', prefix_sep='_')
passenger_count_test = pd.get_dummies(test['passenger_count'], prefix='pc', prefix_sep='_')
store_and_fwd_flag_train = pd.get_dummies(train['store_and_fwd_flag'], prefix='sf', prefix_sep='_')
store_and_fwd_flag_test = pd.get_dummies(test['store_and_fwd_flag'], prefix='sf', prefix_sep='_')
cluster_pickup_train = pd.get_dummies(train['pickup_cluster'], prefix='p', prefix_sep='_')
cluster_pickup_test = pd.get_dummies(test['pickup_cluster'], prefix='p', prefix_sep='_')
cluster_dropoff_train = pd.get_dummies(train['dropoff_cluster'], prefix='d', prefix_sep='_')
cluster_dropoff_test = pd.get_dummies(test['dropoff_cluster'], prefix='d', prefix_sep='_')

month_train = pd.get_dummies(train['Month'], prefix='m', prefix_sep='_')
month_test = pd.get_dummies(test['Month'], prefix='m', prefix_sep='_')
dom_train = pd.get_dummies(train['Day'], prefix='dom', prefix_sep='_')
dom_test = pd.get_dummies(test['Day'], prefix='dom', prefix_sep='_')
hour_train = pd.get_dummies(train['Hour'], prefix='h', prefix_sep='_')
hour_test = pd.get_dummies(test['Hour'], prefix='h', prefix_sep='_')

train.info()

vendor_train.shape,vendor_test.shape

passenger_count_train.shape,passenger_count_test.shape

dom_train.shape,dom_test.shape

passenger_count_train.shape,passenger_count_test.shape

from collections import Counter
Counter(passenger_count_test) - Counter(passenger_count_train)

passenger_count_test = passenger_count_test.drop('pc_9',axis=1)

train.columns

train = train.drop(['id','vendor_id','passenger_count','store_and_fwd_flag','Month','Day','Hour',
                   'pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],axis = 1)

Test_id = test['id']

test = test.drop(['id','vendor_id','passenger_count','store_and_fwd_flag','Month','Day','Hour',
                   'pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'], axis = 1)

train = train.drop(['dropoff_datetime','h_average_speed','m_avg_speed','pickup_lat_bin','pickup_long_bin','trip_duration'], axis = 1)

train.shape,test.shape

Train_Master = pd.concat([train,
                          vendor_train,
                          passenger_count_train,
                          store_and_fwd_flag_train,
                          cluster_pickup_train,
                          cluster_dropoff_train,
                         month_train,
                         dom_train,
                          hour_test
                         ], axis=1)

Test_master = pd.concat([test, 
                         vendor_test,
                         passenger_count_test,
                         store_and_fwd_flag_test,
                         cluster_pickup_test,
                         cluster_dropoff_test,
                         month_test,
                         dom_test,
                         hour_test
                          ], axis=1)

Train_Master.shape,Test_master.shape

Train_Master = Train_Master.drop(['pickup_datetime','pickup_date'],axis = 1)
Test_master = Test_master.drop(['pickup_datetime','pickup_date'],axis = 1)

Train_Master = Train_Master.drop(['pickup_datetime_converted'],axis = 1)
Test_master = Test_master.drop(['pickup_datetime_converted'],axis = 1)

Train, Test = train_test_split(Train_Master[0:100000], test_size = 0.3)

X_train = Train.drop(['log_trip_duration'], axis=1)
Y_train = Train["log_trip_duration"]
X_test = Test.drop(['log_trip_duration'], axis=1)
Y_test = Test["log_trip_duration"]

Y_test = Y_test.reset_index().drop('index',axis = 1)
Y_train = Y_train.reset_index().drop('index',axis = 1)

dtrain = xgb.DMatrix(X_train, label=Y_train)
dvalid = xgb.DMatrix(X_test, label=Y_test)
dtest = xgb.DMatrix(Test_master)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

xgb_pars = {'min_child_weight': 1, 'eta': 0.5, 'colsample_bytree': 0.9, 
            'max_depth': 6,
'subsample': 0.9, 'lambda': 1., 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,
'eval_metric': 'rmse', 'objective': 'reg:linear'}
model = xgb.train(xgb_pars, dtrain, 10, watchlist, early_stopping_rounds=2,
      maximize=False, verbose_eval=1)
print('Modeling RMSLE %.5f' % model.best_score)

xgb.plot_importance(model)

pred = model.predict(dtest)

pred = np.exp(pred) - 1
pred

submission = pd.concat([Test_id, pd.DataFrame(pred)], axis=1)
submission.columns = ['id','trip_duration']
submission['trip_duration'] = submission.apply(lambda x : 1 if (x['trip_duration'] <= 0) else x['trip_duration'], axis = 1)
submission.to_csv("submission_taxi.csv", index=False)



