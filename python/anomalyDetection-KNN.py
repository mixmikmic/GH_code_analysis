get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

from sklearn.neighbors import NearestNeighbors
import matplotlib.font_manager

X_svd = np.loadtxt('../dataset/X_svd.txt', delimiter=',')
X_svd_3 = np.loadtxt('../dataset/X_svd_3.txt', delimiter=',')

nbrs = NearestNeighbors(n_neighbors=20, algorithm='auto').fit(X_svd_3)
distances, indices = nbrs.kneighbors(X_svd_3)
for i in xrange(2):
    print indices[i]
    print distances[i]
    

meanDistances = np.mean(distances, axis=1)[:]
print meanDistances[0]
print np.mean(distances[0])
mean = np.mean(meanDistances)
print mean

def selectMinimumDistanceClusters (indices, distances, threshold):
    finalClusters = []
    normalObservations = []
    anomalies = []
    checked = np.zeros(shape=(len(X_svd_3)))
    for i in xrange(len(meanDistances)):
        if meanDistances[i] >= threshold:
            continue
        else:
            finalClusters.append(i)
            for neighbour in indices[i]:
                if not checked[neighbour]:
                    checked[neighbour] = 1
                    normalObservations.append([neighbour, i])
                else:
                    continue
   
    for point in xrange(len(checked)):
        if not checked[point]:
            anomalies.append(point)
    
    print len(normalObservations), "points clustered"
    print len(anomalies), "anomalies found"
    
    return finalClusters, normalObservations, anomalies

threshold = mean/2
finalClusters, normalObservations, anomalies = selectMinimumDistanceClusters(indices, distances, threshold)
i = 2
while (len(anomalies) >= len(normalObservations)/40):
    if i > 49:
        break
    print "\nRefining the threshold for filtering anomalies"
    threshold = mean/2*i
    finalClusters, normalObservations, anomalies = selectMinimumDistanceClusters(indices, distances, threshold)
    i += 1
print len(finalClusters), "clusters formed"

from operator import itemgetter
print max(normalObservations, key=itemgetter(1))

basemap = plt.cm.get_cmap('hsv', len(X_svd_3))
knnColorMap = np.zeros(shape=(len(X_svd_3), 4))
for item in normalObservations:
    knnColorMap[item[0]] = basemap(item[1])
print len(knnColorMap)

cmap = tuple(map(tuple, knnColorMap))

X_normal = np.zeros(shape=(len(normalObservations), 3))
j = 0
for item in normalObservations:
    temp = []
    for i in xrange(3):
        temp.append(X_svd_3[item[0], i])
    X_normal[j] = temp
    j += 1

X_anomalous = np.zeros(shape=(len(anomalies), 3))
j = 0
for point in anomalies:
    temp = []
    for i in xrange(3):
        #print X_svd_3[point, i]
        temp.append(X_svd_3[point, i])
    X_anomalous[j] = temp
    j += 1
#for i in xrange(5):
#    print X_plot[i, 0], X_plot[i, 1], X_plot[i, 2]
#print X_plot.shape

fig = plt.figure(figsize=(18, 18))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_normal[:,0], X_normal[:,1], X_normal[:,2], s=15, alpha=.3, edgecolor='w', c=cmap)
ax.scatter(X_anomalous[:,0], X_anomalous[:,1], X_anomalous[:,2], s=50, alpha=.8, edgecolor='w', c='blueviolet')

