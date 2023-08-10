# first, read in the data

import os
import csv

os.chdir('../data/')

records = []

with open('call_records.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        records.append(row)

print(records[0]) # print the header
records = records[1:] # remove the header
print(records[0]) # print an example record

# Load up the data that was stored from "1. Feature Engineering"
get_ipython().magic('store -r cph')
get_ipython().magic('store -r num_friends')

from sklearn import preprocessing
import numpy as np

all_numbers = list(set([r[2] for r in records]))

# build the feature vectors
cph = preprocessing.minmax_scale(cph)
num_friends = preprocessing.minmax_scale(num_friends)
data = [[a, b] for (a, b) in zip(cph, num_friends)]
data = np.array(data)
# normalize it - see:
# https://stats.stackexchange.com/questions/21222/are-mean-normalization-and-feature-scaling-needed-for-k-means-clustering
#data = preprocessing.normalize(data)

import matplotlib.pyplot as plt # import our graphing module
plt.rcParams["figure.figsize"] = (10,10) # set the default figure size

plt.scatter(*zip(*data))

plt.xlabel('Average Calls per Hour')
plt.ylabel('Number of Friends')
#plt.title('Histogram of Calls per Hour')
plt.show()

from sklearn.cluster import MeanShift, estimate_bandwidth

# MeanShift requires a bandwidth parameter - you might want to fiddle with it a little to see what happens,
# but it's perhaps also better to just estimate it with math :)
# for detail, see: http://scikit-learn.org/stable/modules/clustering.html#mean-shift
bandwidth = estimate_bandwidth(data, quantile=0.2, n_samples=500)
print(bandwidth)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(data)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = list(set(labels))
n_clusters = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters)


# alternatively, do the clustering with k means
#from sklearn import cluster as clu
#n_clusters = 10
#cluster_output = clu.k_means(data, n_clusters=n_clusters, max_iter=1000)
#labels = cluster_output[1]
#cluster_centers = cluster_output[0]
#print(labels)

# we now have labels, and need to turn that into phone numbers
results = [[] for x in range(n_clusters+1)]
for i,label in enumerate(labels):
    # for each label, grab the number it represents (all_numbers[i]) and
    # put it in the right cluster list (results[label])
    results[label].append(all_numbers[i])
  
for result in results:
    print(result)

get_ipython().magic('store all_numbers')
get_ipython().magic('store labels')

from itertools import cycle

plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(data[my_members, 0], data[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_clusters)
plt.show()

