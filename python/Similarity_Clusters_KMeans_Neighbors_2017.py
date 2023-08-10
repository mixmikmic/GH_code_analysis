# Import the libraries we will be using
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance

get_ipython().magic('matplotlib inline')
import matplotlib.pylab as plt
plt.rcParams['figure.figsize'] = 14, 8

from dstools import data_tools

np.random.seed(36)

data = pd.read_csv("data/scotch.csv")

data = data.drop([u'age', u'dist', u'score', u'percent', u'region', u'district', u'islay', u'midland', u'spey', u'east', u'west', u'north ', u'lowland', u'campbell', u'islands'], axis=1)

data.head()

data_tools.feature_printer(data, 'Aberfeldy') 

data.loc['Aberfeldy']

print ( data_tools.feature_printer(data, 'Bunnahabhain') )


def whiskey_distance(name, distance_measures, n):
    # We want a data frame to store the output
    # distance_measures is a list of the distance measures you want to compute (see below)
    # n is how many "most similar" to report
    distances = pd.DataFrame()
    
    # Find the location of the whiskey we are looking for
    whiskey_location = np.where(data.index == name)[0][0]

    # Go through all distance measures we care about
    for distance_measure in distance_measures:
        # Find all pairwise distances
        current_distances = distance.squareform(distance.pdist(data, distance_measure))
        # Get the closest n elements for the whiskey we care about
        most_similar = np.argsort(current_distances[:, whiskey_location])[0:n]
        # Append results (a new column to the dataframe with the name of the measure)
        distances[distance_measure] = list(zip(data.index[most_similar], current_distances[most_similar, whiskey_location]))
    return distances

whiskey_distance('Bunnahabhain', ['euclidean'], 6)

whiskey_distance('Bunnahabhain', ['euclidean', 'cityblock', 'cosine', 'jaccard'], 10)

data_tools.feature_printer(data, 'Bunnahabhain')

data_tools.feature_printer(data, 'Glenglassaugh')

data_tools.feature_printer(data, 'Benriach')

# This function gets pairwise distances between observations in n-dimensional space.
dists = pdist(data, metric="cosine")

# This scipy's function performs hierarchical/agglomerative clustering on the condensed distance matrix y.
links = linkage(dists, method='average')

# Now we want to plot those 'links' using "dendrogram" function
plt.rcParams['figure.figsize'] = 32, 16

den = dendrogram(links)

plt.xlabel('Samples',fontsize=18)
plt.ylabel('Distance',fontsize=18)
plt.xticks(rotation=90,fontsize=16)
plt.show()

# This function gets pairwise distances between observations in n-dimensional space.
dists = pdist(data, metric="euclidean")

# This scipy's function performs hierarchical/agglomerative clustering on the condensed distance matrix y.
links = linkage(dists, method='average')

# Now we want to plot those 'links' using "dendrogram" function
plt.rcParams['figure.figsize'] = 32, 16

den = dendrogram(links)

plt.xlabel('Samples',fontsize=18)
plt.ylabel('Distance',fontsize=18)
plt.xticks(rotation=90,fontsize=16)
plt.show()

#data[74:75]
#data[1:2]
#data[30:31]
#data[108:109]
#print(np.where(data.index=='Bunnahabhain')[0])
#data[18:19]


k_clusters = 6

## Fit clusters like in our previous models/transformations/standarization 
## (e.g. Logistic, Vectorization,...)

model = KMeans(k_clusters)
model.fit(data)

print ("Records in our dataset (rows): ", len(data.index))
print ("Then we predict one cluster per record, which means length of: ", len(model.predict(data)) )

data.index

clusters = model.predict(data)

clusters

pd.DataFrame(list(zip(data.index,model.predict(data))), columns=['Whiskey','Cluster_predicted']) [0:10]


cluster_listing = {}
for cluster in range(k_clusters):
    cluster_listing['Cluster ' + str(cluster)] = [''] * 109
    where_in_cluster = np.where(clusters == cluster)[0]
    cluster_listing['Cluster ' + str(cluster)][0:len(where_in_cluster)] = data.index[where_in_cluster]

# Print clusters
pd.DataFrame(cluster_listing).loc[0:np.max(np.bincount(clusters)) - 1,:]

## This function returns 2 columns of data and the Y-target

X, Y = data_tools.make_cluster_data()

pd.DataFrame(X).head()

plt.scatter(X[:,0], X[:, 1], s=20)
plt.xlabel("Feature 1",fontsize=18)
plt.ylabel("Feature 2",fontsize=18)
plt.show()

# KMeans
model = KMeans(2)
model.fit(X)

# Predict clusters
clusters = model.predict(X)

# Plot the same points but set two different colors (based on the cluster's results)

plt.scatter(X[:,0], X[:, 1], color=data_tools.colorizer(clusters), linewidth=0, s=20)
plt.xlabel("Feature 1",fontsize=18)
plt.ylabel("Feature 2",fontsize=18)
plt.show()

# KMeans
model = KMeans(3)
model.fit(X)

# Predict clusters
clusters = model.predict(X)

# Plot the same points but now set 3 different colors (based on the cluster's results)

plt.scatter(X[:,0], X[:, 1], color=data_tools.colorizer(clusters), linewidth=0, s=20)
plt.xlabel("Feature 1",fontsize=18)
plt.ylabel("Feature 2",fontsize=18)
plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=.75)

pd.DataFrame( list(zip(X[:,0],X[:, 1],Y_train)), columns=['Feature1','Feature2','Target']) [0:15]


import pylab

plt.scatter( X[:,0], X[:, 1], c = Y, linewidth=0, s=20, cmap = pylab.cm.brg )
plt.xlabel("Feature 1",fontsize=20)
plt.ylabel("Feature 2",fontsize=20)
plt.show()


# KNN
model = KNeighborsClassifier(50)
model.fit(X_train, Y_train)
data_tools.Decision_Surface(X, Y, model, cell_size=.1, surface=True, points=False)


# KNN
model = KNeighborsClassifier(5)
model.fit(X_train, Y_train)
data_tools.Decision_Surface(X, Y, model, cell_size=.05, surface=True, points=False)


# KNN
model = KNeighborsClassifier(5)
model.fit(X_train, Y_train)
data_tools.Decision_Surface(X, Y, model, cell_size=.05, surface=True, points=True)


for k in [1, 10, 50, 100, 1000, 2000]:
    model = KNeighborsClassifier(k)
    model.fit(X_train, Y_train)
    print ("Accuracy with k = %d is %.3f" % (k, metrics.accuracy_score(Y_test, model.predict(X_test))) )
    

for k in [1, 10, 50, 100, 1000]:
    
    model = KNeighborsClassifier(k)
    model.fit(X_train, Y_train)
    probabilities = model.predict_proba(X_test)

    print("KNN with k = %d" % k)
    aucs = 0
    for i in range(5):
        auc = metrics.roc_auc_score(Y_test == i, probabilities[:,i])
        aucs += auc
        print("   AUC for label %d vs. rest = %.3f" % (i, auc))
        
    print("   Average AUC = %.3f\n" % (aucs / 5.0))

