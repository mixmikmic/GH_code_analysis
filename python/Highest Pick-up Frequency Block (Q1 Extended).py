import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
get_ipython().magic('matplotlib inline')

data_2012_01=pd.read_csv("datasets/yellow_tripdata_2012-01.csv")
data_2012_01=data_2012_01.drop("Unnamed: 0",1)
data_2012_01=data_2012_01.drop("passenger_count",1)
data_2012_01=data_2012_01.drop("dropoff_datetime",1)
data_2012_01=data_2012_01.drop("trip_distance",1)
data_2012_01=data_2012_01.drop("dropoff_longitude",1)
data_2012_01=data_2012_01.drop("dropoff_latitude",1)
data_2012_01=data_2012_01.drop("total_amount",1)
data_2012_01.head()

jan_time=pd.to_datetime(pd.Series(data_2012_01["pickup_datetime"]))
jan_time_list=[]
for index in jan_time:
    jan_time_list.append(index.dayofweek)
data_2012_01["weekday"]=jan_time_list
data_2012_01.head()

jan_2012_mon=pd.DataFrame(data_2012_01)
jan_2012_mon=jan_2012_mon.loc[jan_2012_mon["weekday"]==0]
jan_2012_mon=jan_2012_mon.reset_index(drop=True)
print(jan_2012_mon.shape)

jan_2012_mon=jan_2012_mon[jan_2012_mon.pickup_longitude!=0]
jan_2012_mon=jan_2012_mon[jan_2012_mon.pickup_latitude!=0]
jan_2012_mon=jan_2012_mon[jan_2012_mon.pickup_latitude>40]
jan_2012_mon=jan_2012_mon[jan_2012_mon.pickup_latitude<41]
jan_2012_mon=jan_2012_mon[jan_2012_mon.pickup_longitude>-74.5]
jan_2012_mon=jan_2012_mon[jan_2012_mon.pickup_longitude<-72.5]
jan_2012_mon.shape

jan_2012_mon.head(1)

in_range=jan_2012_mon.loc[jan_2012_mon["pickup_longitude"]>=-73.993335-0.018]
in_range=in_range.loc[in_range["pickup_longitude"]<=-73.993335+0.018]
in_range=in_range.loc[in_range["pickup_latitude"]<=40.727717+0.018]
in_range=in_range.loc[in_range["pickup_latitude"]>=40.727717-0.018]
in_range.shape

in_range_time_list=list(in_range["pickup_datetime"])
in_range_hour=[]
for e in in_range_time_list:
    in_range_hour.append(pd.to_datetime(e).hour)
in_range["time"]=in_range_hour
in_range.head()

in_time=in_range.loc[in_range["time"]<21]
in_time=in_time.loc[in_range["time"]>18]
in_time.shape

import matplotlib.path as mplPath

def indexZones(shapeFilename):
    import rtree
    import fiona.crs
    import geopandas as gpd
    index = rtree.Rtree()
    zones = gpd.read_file(shapeFilename).to_crs(fiona.crs.from_epsg(2263))
    for idx,geometry in enumerate(zones.geometry):
        index.insert(idx, geometry.bounds)
    return (index, zones)

def findBlock(p, index, zones):
    match = index.intersection((p.x, p.y, p.x, p.y))
    for idx in match:
        z = mplPath.Path(np.array(zones.geometry[idx].exterior))
        if z.contains_point(np.array(p)):
            return zones['OBJECTID'][idx]
    return -1

def mapToZone(parts):
    import pyproj
    import shapely.geometry as geom
    proj = pyproj.Proj(init="epsg:2263", preserve_units=True)    
    index, zones = indexZones("datasets/block-groups-polygons.geojson")
    i=0
    for line in parts:
        if (line["pickup_longitude"] and line["pickup_latitude"]):
            pickup_location  = geom.Point(proj(float(line["pickup_longitude"]), float(line["pickup_latitude"])))
            try:
                block_id_list.append(findBlock(pickup_location, index, zones))
            except AttributeError:
                drop_list.append(i)
        i+=1

drop_list=[]
block_id_list=[]

mapToZone(in_time.T.to_dict().values())

print((len(drop_list)+len(block_id_list))==in_time.shape[0])

in_time=in_time.drop(in_time.index[drop_list])
in_time["block_id"]=block_id_list
in_time=in_time.reset_index(drop=True)
in_time=in_time.loc[in_time["block_id"]!=-1]
in_time.head()

d1=pd.read_csv("datasets/2012-1-0-id-78.csv")
d2=pd.read_csv("datasets/2012-2-0-id-78.csv")
d3=pd.read_csv("datasets/2012-3-0-id-78.csv")
d4=pd.read_csv("datasets/2012-4-0-id-78.csv")
d5=pd.read_csv("datasets/2012-5-0-id-78.csv")
d6=pd.read_csv("datasets/2012-6-0-id-78.csv")
d7=pd.read_csv("datasets/2012-7-0-id-78.csv")
d8=pd.read_csv("datasets/2012-8-0-id-78.csv")
d9=pd.read_csv("datasets/2012-9-0-id-78.csv")
d10=pd.read_csv("datasets/2012-10-0-id-78.csv")
d11=pd.read_csv("datasets/2012-11-0-id-78.csv")
d12=pd.read_csv("datasets/2012-12-0-id-78.csv")

combine=[d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12]
result = pd.concat(combine)

from sklearn.cluster import KMeans
plon=result['pickup_longitude'].values
plat=result['pickup_latitude'].values
coodinates=np.array([[plon[i],plat[i]] for i in range(len(plon))])
kmeans_n = KMeans(n_clusters=333,  n_init=1)
kmeans_n.fit(coodinates)
labels = kmeans_n.labels_
result["label"]=labels

ls=result.groupby('label').size()
ls=np.array([[ls[i]] for i in range(len(ls))])
lc=kmeans_n.cluster_centers_
train_s=int(len(ls)*0.8)
test_s=int(len(ls)*0.2)
train_f=ls[:train_s]
train_r=lc[:train_s]
test_f=ls[test_s:]
test_r=lc[test_s:]

from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


def fit_model(X, y):
    model = Pipeline([('poly', PolynomialFeatures(degree=3)),
                ('linear', LinearRegression(fit_intercept=False))])
    model.fit(X, y)
    return model

def score_model(model, X, y, Xv, yv):
    return tuple([model.score(X, y), model.score(Xv, yv)])

def fit_model_and_score(data, response, validation, val_response):
    model = fit_model(data, response)
    return score_model(model, data, response, validation, val_response)

print (fit_model_and_score(train_f, train_r,
                           test_f, test_r))

X=ls
y=lc

model=Pipeline([('poly', PolynomialFeatures(degree=3)),
                ('linear', LinearRegression(fit_intercept=False))])
model.fit(X, y)
X_pred=(max(ls))
y_pred=model.predict(X_pred)
y_pred

original=[[-73.993335,40.727717]]
from pygeocoder import Geocoder
predicted_location = Geocoder.reverse_geocode(original[0][1], original[0][0])
# predicted_location.street_address might not be applicable
if predicted_location.street_address:
    predicted_address = predicted_location.street_address
else:
    predicted_address = '%s %s, %s' %(predicted_location.street_number, predicted_location.route, predicted_location.city)
print (predicted_address)

from pygeocoder import Geocoder
predicted_location = Geocoder.reverse_geocode(y_pred[0][1], y_pred[0][0])
# predicted_location.street_address might not be applicable
if predicted_location.street_address:
    predicted_address = predicted_location.street_address
else:
    predicted_address = '%s %s, %s' %(predicted_location.street_number, predicted_location.route, predicted_location.city)
print (predicted_address)

