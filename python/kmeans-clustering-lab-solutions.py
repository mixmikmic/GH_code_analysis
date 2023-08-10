get_ipython().magic('matplotlib inline')

import pandas as pd
import numpy as np
from sklearn import cluster
from sklearn import metrics
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot') 

import seaborn as sns

seeds = pd.read_csv("../datasets/seeds.csv")

# Taking a peek
seeds.head()

# Plot the Data to see the distributions/relationships
import seaborn as sns

# Plot without the "species" hue.
sns.pairplot(seeds)

# Check for nulls
seeds.isnull().sum()
# there is a value for every position in the DF

# Look at the real species labels.
sns.pairplot(data=seeds, hue='species')
# classes appear to have a similar number of samples.
# Blue consistently looks like the divisor between the green and red classes.

seeds.species.value_counts()
# all classes are equally distributed. 

# Check datatypes
seeds.dtypes
# We got an odd-ball, that species guy.

# drop 'species', which is currently acting as a target (categorical)
X = seeds.drop('species', axis = 1)
y = seeds.species

from sklearn.cluster import KMeans

# 2 Clusters
k_mean = KMeans()
k_mean.fit(X)

# Labels and centroids for 8 Clusters
labels = k_mean.labels_
print labels
clusters = k_mean.cluster_centers_
clusters

from sklearn.metrics import silhouette_score

silhouette_score(X, labels)

# Considering silhouette is on a scale of -1 to 1, 0.35 isnt too bad.

# visually examine the cluster that have been created
X_8 = seeds.drop('species', axis=1)
X_8['clusters']=labels

sns.pairplot(data=X_8, hue='clusters')

import random

random.randint(1,25), random.randint(1,25)

# 4 Clusters
k_mean4 = KMeans(n_clusters=4)
k_mean4.fit(X)
labels_4 = k_mean4.labels_
silhouette_score(X, labels_4)

X_4 = seeds.drop('species', axis=1)
X_4['clusters']=labels_4

sns.pairplot(data=X_4, hue='clusters')

# k=4 was the best performing of the Ks i tested
# looks like scatter plot of perimeter vs. asymmetry_coeff
# distingusihed the cluster the best.

# 6 Clusters
k_mean6 = KMeans(n_clusters=6)
k_mean6.fit(X)
labels_6 = k_mean6.labels_
silhouette_score(X, labels_6)

X_6 = seeds.drop('species', axis=1)
X_6['clusters']=labels_6

sns.pairplot(data=X_6, hue='clusters')

# perimeter vs asymmetry_coeff & area vs. asymmetry_coeff
# distiguished the clusters the best visually.

#necessary processing imports
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler

# create dataframe to append info too
results = pd.DataFrame(columns = ['k','silhouette','processing'])


def cluster(ran, data, version):
    for k in ran:
        k_means = KMeans(n_clusters=k)
        k_means.fit(data)
        labels = k_means.labels_
        score = silhouette_score(data, labels)
        results.loc[len(results)]=['c'+str(k), score, version]

def opt_cluster(ran, data):
    cluster(ran, data, 'default')
    
    # normalized version
    Xn = normalize(data)
    cluster(ran, Xn, 'normalized')
    
    # standard scale version
    SS = StandardScaler()
    Xs = SS.fit_transform(data)
    cluster(ran, Xs, 'standard_scaler')
    
    # minmax scale version
    MM = MinMaxScaler()
    Xmm = MM.fit_transform(data)
    cluster(ran, Xmm, 'min_max_scaler')

    return results.loc[results['silhouette'].idxmax()]

ran = range(2,12)

opt_cluster(ran,X)

# build the model with the found optimal parameters
k_mean_opt = KMeans(n_clusters=2)
k_mean_opt.fit(X)
labels_opt = k_mean_opt.labels_

# no preprocessing required since default was the highest silouette
X_opt = seeds.drop('species', axis=1)

X_opt['clusters']=labels_opt
sns.pairplot(data=X_opt, hue='clusters')



