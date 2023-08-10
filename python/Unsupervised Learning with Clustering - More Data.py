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

all_numbers = list(set([r[2] for r in records]))

def calc_calls_per_hour(records):
    call_times = [int(r[0]) for r in records]
    first_call = min(call_times)
    last_call = max(call_times)
    total_calls = len(call_times)
    active_duration = last_call - first_call
    calls_per_hour = total_calls / active_duration
    return calls_per_hour

cph = []
for number in all_numbers:
    n_calls_hour = calc_calls_per_hour([r for r in records if r[2] == number])
    cph.append(n_calls_hour)

import matplotlib.pyplot as plt # import our graphing module
plt.rcParams["figure.figsize"] = (15,5) # set the default figure size

n, bins, patches = plt.hist(cph, 25)

plt.xlabel('Average Calls per Hour')
plt.ylabel('Count')
plt.title('Histogram of Calls per Hour')
plt.show()

def calc_num_friends(records):
    recipients = [r[3] for r in records]
    uniq_recipients = sorted(list(set(recipients)))
    freq = [recipients.count(x) for x in uniq_recipients]
    average_freq = sum(freq) / len(freq)
    friends = [x for (x,y) in zip(uniq_recipients, freq) if y > average_freq]
    return len(friends)

num_friends = []
for num in all_numbers:
    num_friends.append(calc_num_friends([r for r in records if r[2] == num]))

plt.scatter(cph, num_friends)

plt.xlabel('Average Calls per Hour')
plt.ylabel('Number of Friends')
plt.show()

from sklearn import preprocessing
import numpy as np

get_ipython().magic('store -r similarity_matrix')

# build the feature vectors
cph = preprocessing.minmax_scale(cph)
num_friends = preprocessing.minmax_scale(num_friends)

data = [[a, b, *c] for (a, b, c) in zip(cph, num_friends, similarity_matrix)]
data = np.array(data)

plt.scatter(*zip(*data))

plt.xlabel('Average Calls per Hour')
plt.ylabel('Number of Friends')
#plt.title('Histogram of Calls per Hour')
plt.show()

from sklearn import decomposition

# reduce our 1175-dimensional model to a 20-dimensional model
# see: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA
reduced_data = decomposition.PCA(n_components=3).fit_transform(data)

from sklearn.cluster import MeanShift, estimate_bandwidth

bandwidth = estimate_bandwidth(reduced_data, quantile=0.5, n_samples=500)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(reduced_data)
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

