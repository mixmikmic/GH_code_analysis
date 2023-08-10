# Import the libraries we will be using
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance

get_ipython().magic('matplotlib inline')
import matplotlib.pylab as plt
plt.rcParams['figure.figsize'] = 14, 8

# I've abstracted some of what we'll be doing today into a library.
# You can take a look at this code if you want by going into `dstools/data_tools.py`
from dstools import data_tools

np.random.seed(36)

data = pd.read_csv("data/scotch.csv")
data = data.drop([u'age', u'dist', u'score', u'percent', u'region', u'district', u'islay', u'midland', u'spey', u'east', u'west', u'north ', u'lowland', u'campbell', u'islands'], axis=1)

data.head()

print data_tools.feature_printer(data, 'Bunnahabhain')

# For a full list of the distance metrics supported by scipy, check:
# http://docs.scipy.org/doc/scipy/reference/spatial.distance.html

def whiskey_distance(name, distance_measures, n):
    # We want a data frame to store the output
    distances = pd.DataFrame()
    
    # Find the location of the whiskey we are looking for
    whiskey_location = np.where(data.index == name)[0][0]

    # Go through all distance measures we care about
    for distance_measure in distance_measures:
        # Find all pairwise distances
        current_distances = distance.squareform(distance.pdist(data, distance_measure))
        # Get the closest n for the whiskey we care about
        most_similar = np.argsort(current_distances[:, whiskey_location])[0:n]
        # Append results
        distances[distance_measure] = zip(data.index[most_similar], current_distances[most_similar, whiskey_location])
    return distances

whiskey_distance('Bunnahabhain', ['euclidean'], 6)

whiskey_distance('Bunnahabhain', ['euclidean', 'cityblock', 'cosine'], 6)

print data_tools.feature_printer(data, 'Bunnahabhain')

print data_tools.feature_printer(data, 'Bruichladdich')

print data_tools.feature_printer(data, 'Ardberg')

# This function gets pairwise distances between observations in n-dimensional space.
dists = pdist(data, metric="cosine")

# This function performs hierarchical/agglomerative clustering on the condensed distance matrix y.
links = linkage(dists)

# Now we want to plot the dendrogram
plt.rcParams['figure.figsize'] = 32, 16
den = dendrogram(links)
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.xticks(rotation=45)
plt.show()
plt.rcParams['figure.figsize'] = 10, 8

k_clusters = 6

# Fit and predict clusters
model = KMeans(k_clusters)
model.fit(data)
clusters = model.predict(data)

# Do some messy stuff to print a nice table of clusters
cluster_listing = {}
for cluster in range(k_clusters):
    cluster_listing['Cluster ' + str(cluster)] = [''] * 109
    where_in_cluster = np.where(clusters == cluster)[0]
    cluster_listing['Cluster ' + str(cluster)][0:len(where_in_cluster)] = data.index[where_in_cluster]

# Print clusters
pd.DataFrame(cluster_listing).loc[0:np.max(np.bincount(clusters)) - 1,:]

X, Y = data_tools.make_cluster_data()

plt.scatter(X[:,0], X[:, 1], s=20)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# KMeans
model = KMeans(2)
model.fit(X)
clusters = model.predict(X)
plt.scatter(X[:,0], X[:, 1], color=data_tools.colorizer(clusters), linewidth=0, s=20)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=.75)

# KNN
model = KNeighborsClassifier(10)
model.fit(X_train, Y_train)
data_tools.Decision_Surface(X, Y, model, cell_size=.05, surface=True, points=False)

for k in [1, 10, 100]:
    model = KNeighborsClassifier(k)
    model.fit(X_train, Y_train)
    print "Accuracy with k = %d is %.3f" % (k, metrics.accuracy_score(Y_test, model.predict(X_test)))

for k in [1, 10, 100]:
    model = KNeighborsClassifier(k)
    model.fit(X_train, Y_train)
    probabilities = model.predict_proba(X_test)

    print "KNN with k = %d" % k
    aucs = 0
    for i in range(5):
        auc = metrics.roc_auc_score(Y_test == i, probabilities[:,i])
        aucs += auc
        print "   AUC for label %d vs. rest = %.3f" % (i, auc)
    print "   Average AUC = %.3f\n" % (aucs / 5.0)

