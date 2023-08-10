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
def calculate_cluster_means(X, labels, quiet=False):
    lbls = np.unique(labels)

    cluster_means = [np.mean(X[labels==num, :], axis=0) for num in lbls if num != -1]
    
    if not quiet:
        print "Cluster labels: {}".format(np.unique(lbls))
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
    
def plot_the_clusters(X, dbscan_model, noise_point=None, set_size=True, 
                      markersize=14):
    labels = dbscan_model.labels_
    clusters = len(set(labels))
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    if set_size:
        plt.subplots(figsize=(12,8))

    for k, c in zip(unique_labels, colors):
        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=c,
                markeredgecolor='k', markersize=markersize)
        
    if noise_point is not None:
        plt.plot(noise_point[0], noise_point[1], 'xr', markersize=markersize+3)

    plt.title("Total Clusters: {}".format(clusters), fontsize=14, y=1.01)
    
    return colors, unique_labels
    
def do_yo_thang(X, dbscan_model, noise_point):
    cluster_means = calculate_cluster_means(X, dbscan_model.labels_)
    print_3_distances(noise_point, cluster_means)
    return plot_the_clusters(X, dbscan_model, noise_point)

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

noisy_prices2 = prices + np.random.normal(0, 75, 60)
noisy_prices2[30] = 500

np2_mm175 = MinMaxScaler(feature_range=(-1.75, 1.75)).fit_transform(noisy_prices2[:, None])
d_mm3 = MinMaxScaler(feature_range=(-3,3)).fit_transform(days[:, None])
X2_mm175mm3 = makeX(d_mm3, np2_mm15)

db = DBSCAN(eps=0.45, min_samples=4).fit(X2_mm175mm3)
noise_point = X2_mm175mm3[30, :]
do_yo_thang(X2_mm175mm3, db, noise_point)
for index, xy in enumerate(zip(d_mm3, np2_mm15)):
    plt.annotate('{}: ({:0.2f}, {:0.2f})'.format(index, xy[0][0], xy[1][0]), xytext=(xy[0]-0.25, xy[1]-0.1), xy=xy)
# What do we not want to happen? How can we control epsilon to achieve that?

get_value = lambda x: x[1]

def axes_plot_the_clusters(X, dbscan_model, noise_point, ax, markersize=14):
    labels = dbscan_model.labels_
    clusters = len(set(labels))
    unique_labels = list(set(labels))
    cluster_means = calculate_cluster_means(X, labels, quiet=True)
    
    if -1 not in unique_labels:
        unique_labels = [-1] + unique_labels
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    for k, c in zip(unique_labels, colors):
        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        for array in xy:
            x, y = array
            if k == -1 and len(unique_labels) > 1:
                closest_cluster = min(enumerate([euclidean(array, cm) for cm in cluster_means]), key=get_value)[0]
                ax.plot(x, y, 'o', markerfacecolor='none', markeredgecolor=colors[closest_cluster], 
                        mew=3, markersize=markersize)
            else:
                ax.plot(x, y, 'o', markerfacecolor=c, markeredgecolor='k', markersize=markersize)
        
    ax.plot(noise_point[0], noise_point[1], 'xr', markersize=markersize+3)
    
    return colors, unique_labels

plt.subplots

columns = 5
rows = 25
multiple = 20/columns

fig, subplots = plt.subplots(rows, columns, sharex='col', sharey='row',
                            figsize=(columns*multiple,
                                     rows*multiple
                                    ))
label_fontsize = 18

minPts_range = np.linspace(2, 6, columns)
eps_range = np.linspace(0.1, 0.6, rows)

place = 1
for r_index, e in enumerate(eps_range):
    for c_index, m in enumerate(minPts_range):
        ax = subplots[r_index, c_index]
        db = DBSCAN(eps=e, min_samples=m).fit(X2_mm175mm3)
        num_actual_clusters = len([c for c in np.unique(db.labels_) if c != -1])
        num_outliers = len(X2_mm175mm3[db.labels_ == -1])
        axes_plot_the_clusters(X2_mm175mm3, db, noise_point, ax, markersize=8)
        ax.set_title("Number of Clusters: {}\nOutliers: {}".format(num_actual_clusters, num_outliers), 
                     fontsize=label_fontsize, y=1.01)
        ax.yaxis.set_label_position('right')
        ax.yaxis.labelpad = label_fontsize - 2
        ax.set_ylabel('eps: {:.2f}, minPts: {}'.format(e, int(m)), 
                      fontdict={'fontsize': label_fontsize - 2}, rotation=270)
plt.tight_layout()

get_ipython().magic('debug')

s = subplots[0,0]
s.get_title()

