import pandas as pd
import numpy as np
from geopy.distance import great_circle
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

distances = pd.read_csv("distancias_representativas_solar.csv")
substations = pd.read_csv("subestaciones_w_lz.csv")
del substations['Unnamed: 0']
del substations['id']
substations = substations.sort_values(by = ['lz'], ascending = [True])

# First, we get all the existing substations in the loadzone.
X = []
Y = []
sub_lz_1 = []

for i, row in substations.iterrows():
    if (row['lz'] == 1.0):
        sub_lz_1.append((row['X'], row['Y']))
        X.append(row['X'])
        Y.append(row['Y'])
        
df = pd.DataFrame (index = [i for i in range(0, len(X))], columns = ['X', 'Y'])
df['X'] = X
df['Y'] = Y

# We can see the location of each substation as a point in the plane. Thus, if we set the known representative 
# distance of each particular loadzone as a radius around each of the points (i.e., each of the substations),
# we will have several circles inside a loadzone, each of which has radius r, being r the representative distance
# of the loadzone (distances.iloc[0]['km']).

# Using DBSCAN clustering, we get the subclusters inside a loadzone. Note that a single substation can itself be a
# subcluster; it wouldn't be beneficial to ignore this substations since very different energy-generation
# profiles can exists among the same loadzone and we want to take them into account.
# https://en.wikipedia.org/wiki/DBSCAN

distance_matrix = squareform(pdist(sub_lz_1, (lambda u, v: great_circle(u, v).kilometers)))
db = DBSCAN(eps = 2 * distances.iloc[0]['km'] + 1, min_samples = 1, metric = 'precomputed')
y_db = db.fit_predict(distance_matrix)
cluster = y_db

df['cluster'] = cluster
df = df.drop(df[df.cluster == -1].index)

plt.scatter(df['X'], df['Y'], c = df['cluster'])
plt.show()

centroids = []

i = 0
while True:
    _df = df.loc[df['cluster'] == i]
    
    if _df.empty:
        break
    centroids.append((((sum(_df['X'])/len(_df.index)), (sum(_df['Y']))/len(_df.index)), i))
    i = i + 1
    
centroids

centroids = []

for s in range(1, 54):
    X = []
    Y = []
    sub_lz = []

    for i, row in substations.iterrows():
        if (row['lz'] == s):
            sub_lz.append((row['X'], row['Y']))
            X.append(row['X'])
            Y.append(row['Y'])
        
    df = pd.DataFrame (index = [i for i in range(0, len(X))], columns = ['X', 'Y'])
    df['X'] = X
    df['Y'] = Y
    
    distance_matrix = squareform(pdist(sub_lz, (lambda u, v: great_circle(u, v).kilometers)))
    db = DBSCAN(eps = 2 * distances.iloc[s-1]['km'] + 1, min_samples = 1, metric = 'precomputed')
    y_db = db.fit_predict(distance_matrix)
    cluster = y_db

    df['cluster'] = cluster
    df = df.drop(df[df.cluster == -1].index)

    i = 0
    while True:
        _df = df.loc[df['cluster'] == i]
    
        if _df.empty:
            break
        
        centroids.append(((sum(_df['X'])/len(_df.index)), (sum(_df['Y']))/len(_df.index), s, i))
        
        i = i + 1

centroids



