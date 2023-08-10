import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
get_ipython().magic('matplotlib inline')

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
result=result.drop("Unnamed: 0",1)
result.head()

loc=result.loc[result["block_id"]==9052]
loc_lon=loc["pickup_longitude"].mean()
loc_lat=loc["pickup_latitude"].mean()

lon=result["pickup_longitude"].values
lat=result["pickup_latitude"].values
plt.figure(figsize=(15,11))
plt.scatter(lon,lat,alpha=0.1,s=30)
original=[[-73.993335,40.727717]]
predicted=[[-73.99683239, 40.73451787]]
plt.scatter(original[0][0],original[0][1],c='r',s=111)
plt.scatter(predicted[0][0],predicted[0][1],c='g',s=111)
plt.scatter(loc_lon,loc_lat,c='y',s=111)
plt.xlabel("longitude")
plt.ylabel("latitude")
plt.show()
# red: current location
# yellow: actual location based on actual data
# green: predicted location using regression

temp=result.groupby("block_id").count()
counted=list(temp["time"].values)
bid=list(temp.index.values)
final=pd.DataFrame(counted,bid)
final.columns = ["pickup_amount"]
final.head()

plt.bar(list(final.index.values),final["pickup_amount"])
plt.show()

max_t=0
index=0
mark=0
for e in counted:
    if e>max_t:
        max_t=e
        mark=index
    index+=1
print("block "+str(bid[mark])+": "+str(counted[34])+" total pickups")

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
        pickup_location  = geom.Point(proj(float(line["pickup_longitude"]), float(line["pickup_latitude"])))
        try:
            print(findBlock(pickup_location, index, zones))
        except AttributeError:
            break
        i+=1

plon=[-73.99683239]
plat=[40.73451787]
to_id=pd.DataFrame({'pickup_longitude':plon, 'pickup_latitude':plat})
mapToZone(to_id.T.to_dict().values())

original=[[loc_lon,loc_lat]]
from pygeocoder import Geocoder
predicted_location = Geocoder.reverse_geocode(original[0][1], original[0][0])
# predicted_location.street_address might not be applicable
if predicted_location.street_address:
    predicted_address = predicted_location.street_address
else:
    predicted_address = '%s %s, %s' %(predicted_location.street_number, predicted_location.route, predicted_location.city)
print (predicted_address)

