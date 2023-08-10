from __future__ import division
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.vq import kmeans,vq

# data generation
data = np.vstack((np.random.rand(150,2) + np.array([.5,.5]), np.random.rand(150,2)))
plt.plot(data[:, 0], data[:, 1], 'xg')
plt.show()

centers = np.array([[0.2, 0.2], [0.2, 0.1], [0.2, 0.4]])
idx2,_ = vq(data, centers)

# some plotting using numpy's logical indexing
plt.plot(data[idx2==0,0], data[idx2==0,1],'ob',
         data[idx2==1,0], data[idx2==1,1],'or',
         data[idx2==2,0], data[idx2==2,1],'oy')

plt.plot(centers[:,0], centers[:,1], 'sg',markersize=8)
plt.title('Initial state')
plt.show()

c1 = data[idx2==0,:].mean(axis=0)
c2 = data[idx2==1,:].mean(axis=0)
c3 = data[idx2==2,:].mean(axis=0)

print 'Mean for cluster 1', c1
print 'Mean for cluster 2', c2
print 'Mean for cluster 3', c3

ncenter = np.vstack((c1, c2, c3))

idx2,_ = vq(data, ncenter)

# some plotting using numpy's logical indexing
plt.plot(data[idx2==0,0], data[idx2==0,1],'ob',
         data[idx2==1,0], data[idx2==1,1],'or',
         data[idx2==2,0], data[idx2==2,1],'oy')

plt.plot(ncenter[:,0], ncenter[:,1],'sg',markersize=8)
plt.title('Iteration #1')
plt.show()

c1 = data[idx2==0,:].mean(axis=0)
c2 = data[idx2==1,:].mean(axis=0)
c3 = data[idx2==2,:].mean(axis=0)

print 'Mean for cluster 1', c1
print 'Mean for cluster 2', c2
print 'Mean for cluster 3', c3

ncenter = np.vstack((c1, c2, c3))

idx2,_ = vq(data, ncenter)

# some plotting using numpy's logical indexing
plt.plot(data[idx2==0,0], data[idx2==0,1],'ob',
         data[idx2==1,0], data[idx2==1,1],'or',
         data[idx2==2,0], data[idx2==2,1],'oy')

plt.plot(ncenter[:,0], ncenter[:,1],'sg',markersize=8)
plt.title('Iteration #2')
plt.show()

# computing K-Means with K = 2 (2 clusters)
centroids, distortion = kmeans(data, 3)
# assign each sample to a cluster
idx, J = vq(data,centroids)

# some plotting using numpy's logical indexing
plt.plot(data[idx==0,0],data[idx==0,1],'ob',
         data[idx==1,0],data[idx==1,1],'or',
         data[idx==2,0],data[idx==2,1],'oy')
plt.plot(centroids[:,0],centroids[:,1],'sg',markersize=8)
plt.show()

data[0, :], idx[0], J[0]

print 'Kmeans distortion (cost)', distortion 

costs_x_iter = []
for iter in [10, 20, 30, 40, 50]:
    _, cost = kmeans(data, 2, iter=iter)
    costs_x_iter.append([iter, cost])
    
i, c = zip(*costs_x_iter)
plt.plot(i, c)
plt.show()

print 'Kmeans distortion (cost) with 30 iterations', cost

costs_x_k = []
for k in xrange(2, 30):
    _, cost = kmeans(data, k)
    costs_x_k.append([k, cost])
    
k, c = zip(*costs_x_k)
plt.plot(k, c)
plt.show()

from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits

digits_data = load_digits()

np.random.seed(42)
centroids, _ = kmeans(digits_data.data, 10)

# assign each sample to a cluster
idx, _ = vq(digits_data.data, centroids)

plt.imshow(centroids[6].reshape(8, 8), cmap='gray')

loc = np.where(digits_data.target == 2)[0]
y_true = digits_data.target[loc] 
y_pred = idx[idx == 6]
print '{:.2f}%'.format(100 * accuracy_score(y_true[:100], y_pred[:100]))

loc[0], digits_data.target[2], idx[2]

np.where(idx == 2)[0][:3]

digits_data.target[11]



