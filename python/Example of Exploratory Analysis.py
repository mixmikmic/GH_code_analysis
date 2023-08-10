import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, MaxAbsScaler
from sklearn.cluster import DBSCAN

get_ipython().magic('matplotlib inline')

days = np.arange(60)
prices1 = np.random.normal(0, 35, size=20) + 400
prices2 = np.random.normal(0, 35, size=20) + 800
prices3 = np.random.normal(0, 35, size=20) + 400

prices = np.concatenate([prices1, prices2, prices3], axis=0)

print prices.shape
days.shape

X = np.concatenate([days[:, None], prices[:, None]], axis=1)

plt.scatter(days, prices)

# create a test point
print prices[30]
prices[30] = 652

plt.scatter(days, prices)
plt.plot(30, 652, 'or')

# run dbscan
X = np.concatenate([days[:, None], prices[:, None]], axis=1)
db = DBSCAN(eps=30, min_samples=5).fit(X)

# shamelessly copied code below ;)
labels = db.labels_
clusters = len(set(labels))
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
 
plt.subplots(figsize=(12,8))
 
for k, c in zip(unique_labels, colors):
    class_member_mask = (labels == k)
    xy = X[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=c,
            markeredgecolor='k', markersize=14)
 
plt.title("Total Clusters: {}".format(clusters), fontsize=14,
        y=1.01)

from scipy.spatial.distance import euclidean, chebyshev, cityblock

# get the distances to each clusters
labels = db.labels_
lbls = np.unique(db.labels_)
print "Cluster labels: {}".format(np.unique(lbls))

cluster_means = [np.mean(X[labels==num, :], axis=0) for num in range(lbls[-1] + 1)]
print "Cluster Means: {}".format(cluster_means)

noise_point = X[30, :]

# euclidean
dist = [euclidean(noise_point, cm) for cm in cluster_means]
print "Euclidean distance: {}".format(dist)

# chebyshev
dist = [chebyshev(noise_point, cm) for cm in cluster_means]
print "Chebysev distance: {}".format(dist)

# cityblock
dist = [cityblock(noise_point, cm) for cm in cluster_means]
print "Cityblock (Manhattan) distance: {}".format(dist)

# let's create some helper functions
def calculate_cluster_means(X, labels):
    lbls = np.unique(labels)
    print "Cluster labels: {}".format(np.unique(lbls))

    cluster_means = [np.mean(X[labels==num, :], axis=0) for num in range(lbls[-1] + 1)]
    print "Cluster Means: {}".format(cluster_means)
    return cluster_means
    
def print_3_distances(noise_point, cluster_means):
    # euclidean
    dist = [euclidean(noise_point, cm) for cm in cluster_means]
    print "Euclidean distance: {}".format(dist)

    # chebyshev
    dist = [chebyshev(noise_point, cm) for cm in cluster_means]
    print "Chebysev distance: {}".format(dist)

    # cityblock
    dist = [cityblock(noise_point, cm) for cm in cluster_means]
    print "Cityblock (Manhattan) distance: {}".format(dist)
    
def plot_the_clusters(X, dbscan_model, noise_point=None):
    labels = dbscan_model.labels_
    clusters = len(set(labels))
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    plt.subplots(figsize=(12,8))

    for k, c in zip(unique_labels, colors):
        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=c,
                markeredgecolor='k', markersize=14)
        
    if noise_point is not None:
        plt.plot(noise_point[0], noise_point[1], 'xr')

    plt.title("Total Clusters: {}".format(clusters), fontsize=14, y=1.01)
    
def do_yo_thang(X, dbscan_model, noise_point):
    cluster_means = calculate_cluster_means(X, dbscan_model.labels_)
    print_3_distances(noise_point, cluster_means)
    plot_the_clusters(X, dbscan_model, noise_point)

# Let's start playing with scalings
# First we'll do what he did in the book this will have the effect of
# weighting each feature about equally for euclidean distance

X_ss = StandardScaler().fit_transform(X)
db_ss = DBSCAN(eps=0.4, min_samples=3).fit(X_ss)
noise_point = X_ss[30, :]
do_yo_thang(X_ss, db_ss, noise_point)

# Let's make it a little harder now
noisy_prices = prices + np.random.uniform(-100, 100, 60)
noisy_prices[30] = 500
X = np.concatenate([days[:, None], noisy_prices[:, None]], axis=1)

X_ss = StandardScaler().fit_transform(X)
db_ss = DBSCAN(eps=0.4, min_samples=5).fit(X_ss)
noise_point = X_ss[30, :]
do_yo_thang(X_ss, db_ss, noise_point)

# add another helper fxn
def makeX(days, prices):
    return np.concatenate([days, prices], axis=1)

prices_ss = StandardScaler().fit_transform(noisy_prices[:, None])
prices_rob = RobustScaler().fit_transform(noisy_prices[:, None])

days_mm4 = MinMaxScaler(feature_range=(-4,4)).fit_transform(days[:, None])

X_ssmm4 = makeX(days_mm4, prices_ss)
db = DBSCAN(eps=0.6, min_samples=5).fit(X_ssmm4)
# show with X_ss again

noise_point = X_ssmm4[30, :]
do_yo_thang(X_ssmm4, db, noise_point)
for index, xy in enumerate(zip(days_mm4, prices_ss)):
    plt.annotate('{}: ({:0.2f}, {:0.2f})'.format(index, xy[0][0], xy[1][0]), xytext=(xy[0]-0.45, xy[1]-0.15), xy=xy)

print_3_distances(X_ssmm4[50,:], calculate_cluster_means(X_ssmm4, db.labels_))
print
print_3_distances(X_ssmm4[53, :], calculate_cluster_means(X_ssmm4, db.labels_))

X_rbmm4 = makeX(days_mm4, prices_rob)
db = DBSCAN(eps=0.6, min_samples=5).fit(X_ssmm4)
# show with X_ss again

noise_point = X_rbmm4[30, :]
do_yo_thang(X_rbmm4, db, noise_point)
for index, xy in enumerate(zip(days_mm4, prices_rob)):
    plt.annotate('{}: ({:0.2f}, {:0.2f})'.format(index, xy[0][0], xy[1][0]), xytext=(xy[0]-0.45, xy[1]-0.08), xy=xy)

print_3_distances(X_rbmm4[50,:], calculate_cluster_means(X_ssmm4, db.labels_))
print
print_3_distances(X_rbmm4[53, :], calculate_cluster_means(X_ssmm4, db.labels_))



