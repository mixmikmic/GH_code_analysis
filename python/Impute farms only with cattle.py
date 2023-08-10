import numpy as np
import random as random
import matplotlib.pyplot as plt
from matplotlib import animation
import time
import pandas as pd
import multiprocessing
import itertools
from scipy.spatial import distance
np.set_printoptions(threshold=np.nan)
import os
import seaborn as sns
np.random.seed(5761)

os.chdir('/Users/apple/Desktop/group/data');

Kenya_cattle_2006=pd.read_csv('crop_cattle_2006.txt', 
                       names = ["Long", "Lat", "Holding Number"], delim_whitespace=True)

Kenya_cattle_2006

Latitude=Kenya_cattle_2006["Lat"].values
Longtitude=Kenya_cattle_2006["Long"].values
Holding_number=Kenya_cattle_2006["Holding Number"].values

b=list(set(Latitude))
b.sort()
del b[0]
b1=list(set(Latitude))
b1.sort()
del b[-1]
dist_lat=[x - y for x, y in zip(b, b1)]

c=list(set(Longtitude))
c.sort()
del c[0]
c1=list(set(Longtitude))
c1.sort()
del c1[-1]
dist_long=[x - y for x, y in zip(c, c1)]

dlat=np.mean(dist_lat)
print(np.var(dist_lat))
dlong=np.mean(dist_long)
print(np.var(dist_long))
dlat_half=dlat/2
dlong_half=dlong/2

a=Holding_number.reshape((67,93))

ax = sns.heatmap(a,cmap="YlGnBu",xticklabels=False,yticklabels=False)
plt.show()

from math import sin, cos, sqrt, atan2, radians

# approximate radius of earth in km
R = 6373.0

lat1 = radians(0.02916667)
lon1 = radians(35.82083333)
lat2 = radians(0.0375)
lon2 = radians(35.82916667)

dlon = lon2 - lon1
dlat = lat2 - lat1

a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
c = 2 * atan2(sqrt(a), sqrt(1 - a))

distance = R * c

print("Result:", distance,"km")

lat1 = radians(0.0375)
lon1 = radians(35.820833333333326)
lat2 = radians(0.0375)
lon2 = radians(35.829166666666666)

dlon = lon2 - lon1
dlat = lat2 - lat1

a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
c = 2 * atan2(sqrt(a), sqrt(1 - a))

distance = R * c

print("Result:", distance,"km")

Kenya=pd.read_csv("completeData2.csv")

cattle=Kenya['cattle'].values

missing_farm_lat=list()
missing_farm_long=list()
missing_farm_hold=list()

for i in range(1):
    lat_i=Latitude[i]
    long_i=Longtitude[i]
    hold_i=round(Holding_number[i])
    
    
    lat_lower=lat_i-dlat_half
    lat_upper=lat_i+dlat_half
    long_lower=long_i-dlong_half
    long_upper=long_i+dlong_half
    while (True):
        impute_farm_lat=np.random.uniform(lat_lower,lat_upper)
        impute_farm_long=np.random.uniform(long_lower,long_upper)
        impute_farm_hold=np.random.choice(cattle)       
        
        if hold_i>impute_farm_hold:
            missing_farm_lat.append(impute_farm_lat)
            missing_farm_long.append(impute_farm_long)
            missing_farm_hold.append(impute_farm_hold)
        
            hold_i=hold_i-impute_farm_hold    
        else:
            missing_farm_lat.append(impute_farm_lat)
            missing_farm_long.append(impute_farm_long)
            missing_farm_hold.append(int(hold_i))
            break

missing_farm_hold

missing_farm_lat=list()
missing_farm_long=list()
missing_farm_hold=list()

N=len(Latitude)

for i in range(N):
    lat_i=Latitude[i]
    long_i=Longtitude[i]
    hold_i=round(Holding_number[i])
    
    lat_lower=lat_i-dlat_half
    lat_upper=lat_i+dlat_half
    long_lower=long_i-dlong_half
    long_upper=long_i+dlong_half
    if (hold_i!=0):
        while (True):
            impute_farm_lat=np.random.uniform(lat_lower,lat_upper)
            impute_farm_long=np.random.uniform(long_lower,long_upper)
            impute_farm_hold=np.random.choice(cattle)       
            if hold_i>impute_farm_hold:
                missing_farm_lat.append(impute_farm_lat)
                missing_farm_long.append(impute_farm_long)
                missing_farm_hold.append(impute_farm_hold)
                hold_i=hold_i-impute_farm_hold    
            else:
                missing_farm_lat.append(impute_farm_lat)
                missing_farm_long.append(impute_farm_long)
                missing_farm_hold.append(int(hold_i))
                break
    

plt.scatter(missing_farm_long,missing_farm_lat)
plt.show()

latitude=Kenya["lat"].values
longtitude=Kenya["long"].values
cluster=Kenya["X__1"].values

cluster_unique=list(set(cluster))
N_cluster=len(cluster_unique)

plt.scatter(latitude[cluster==cluster_unique[6]],longtitude[cluster==cluster_unique[6]])
plt.show()

index6=[i for i in range(len(cluster)) if cluster[i]==cluster_unique[6] and latitude[i]<-0.2]
cluster[index6]='NA'

plt.scatter(latitude[cluster==cluster_unique[9]],longtitude[cluster==cluster_unique[9]])
plt.show()

index9=[i for i in range(len(cluster)) if cluster[i]==cluster_unique[9] and longtitude[i]>36.04]
cluster[index9]='NA'

plt.scatter(latitude[cluster==cluster_unique[10]],longtitude[cluster==cluster_unique[10]])
plt.show()

index10=[i for i in range(len(cluster)) if cluster[i]==cluster_unique[10] and longtitude[i]>35.9]
cluster[index10]='NA'

delete_index=list()
N_missing=len(missing_farm_lat)

for i in range(N_cluster):
    lat_ci=latitude[cluster==cluster_unique[i]]
    long_ci=longtitude[cluster==cluster_unique[i]]
    min_lat_ci=min(lat_ci)
    max_lat_ci=max(lat_ci)
    min_long_ci=min(long_ci)
    max_long_ci=max(long_ci)
    index=[k for k in range(N_missing) if (missing_farm_lat[k]<max_lat_ci) and (missing_farm_lat[k]>min_lat_ci) and (missing_farm_long[k]>min_long_ci) and (missing_farm_long[k]<max_long_ci)]
    delete_index.extend(index)   

delete_list_unique=list(set(delete_index))
print(len(delete_list_unique))

for i in sorted(delete_list_unique, reverse=True):
    del missing_farm_lat[i]
    del missing_farm_long[i]
    del missing_farm_hold[i]

plt.scatter(missing_farm_long,missing_farm_lat)
#plt.scatter(longtitude,latitude)
plt.show()

df = pd.DataFrame({'lat':missing_farm_lat,'long':missing_farm_long,'cattle':missing_farm_hold})

print (df)

df.to_csv('imputed_farm_cattle', sep='\t', encoding='utf-8')

#readdata=pd.read_csv('imputed_farm_cattle', delim_whitespace=True)

#readdata



