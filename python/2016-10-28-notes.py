from sklearn.datasets import load_digits

digits = load_digits()

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

digits_2D = pca.fit_transform(digits.data)

pc1 = pca.components_[0,:]
pc2 = pca.components_[1,:]

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.subplot(1,2,1)
plt.imshow(pc1.reshape(8,8),cmap='binary')
plt.subplot(1,2,2)
plt.imshow(pc2.reshape(8,8),cmap='binary');

plt.figure(figsize=(18,8))
plt.scatter(digits_2D[:,0],digits_2D[:,1],c=digits.target,s=100,lw=0,alpha=0.5)
plt.colorbar()

N = 8
pca_ND = PCA(n_components=N)
plt.figure(figsize=(15,10))
pca_ND.fit(digits.data)
pc = pca_ND.components_
plt.subplot(1,N,1)
for n in range(0,N):
    plt.subplot(1,N,n+1)
    plt.imshow(pc[n,:].reshape(8,8),cmap='gray')
    plt.axis('off')

ones = digits.data[digits.target == 1]

pca_ones = PCA(n_components=2)
ones_2D = pca_ones.fit_transform(ones)

plt.figure(figsize=(18,8))
plt.scatter(ones_2D[:,0],ones_2D[:,1],s=100)

from sklearn.cluster import KMeans

N = 3
ones_km = KMeans(n_clusters=N)

ones_km.fit(ones)

clusters = ones_km.cluster_centers_

clusters.shape

plt.figure(figsize=(18,10))
plt.subplot(1,N,1)
for n in range(0,N):
    plt.subplot(1,N,n+1)
    plt.imshow(clusters[n,:].reshape(8,8),cmap='binary')
    plt.axis('off')

N = 5
plt.figure(figsize=(10,10))
plt.subplot(10,N,1)
for n in range(0,10):
    this_digit = digits.data[digits.target == n]
    km = KMeans(n_clusters = N)
    km.fit(this_digit)
    kinds_of_this_digit = km.cluster_centers_
    for m in range(0,N):
        plt.subplot(10,N,n*N+m+1)
        plt.imshow(kinds_of_this_digit[m,:].reshape(8,8),cmap='binary')
        plt.axis('off')
plt.show()

fours = digits.data[digits.target == 4]
km = KMeans(n_clusters=3)
km.fit(fours)
targets = km.predict(fours)
pca = PCA(n_components=2)
fours_2D = pca.fit_transform(fours)
plt.figure(figsize=(18,6))
plt.scatter(fours_2D[:,0],fours_2D[:,1],c=targets,s=100,lw=0,alpha=0.8)
plt.colorbar();

plt.subplot(1,3,1)
plt.imshow(fours[0,:].reshape(8,8),cmap='binary')
plt.axis('off')
plt.subplot(1,3,2)
plt.imshow(fours[5,:].reshape(8,8),cmap='binary')
plt.axis('off')
plt.subplot(1,3,3)
plt.imshow(fours[13,:].reshape(8,8),cmap='binary')
plt.axis('off');

