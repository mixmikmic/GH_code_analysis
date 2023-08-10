import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().magic('matplotlib inline')

from sklearn.datasets import load_digits

digits = load_digits()

digits.data.shape

digits.images.shape

plt.imshow(digits.images[100,:,:],cmap='binary')

from sklearn.decomposition import PCA

pca = PCA(n_components=2) # We want 2 principal components so that we can plot the dataset in 2D

pca.fit(digits.data)

pca.components_.shape

p1 = pca.components_[0,:] # First principal component
p2 = pca.components_[1,:] # Second principal component
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.imshow(p1.reshape(8,8),cmap='binary'), plt.title('First Principal Component')
plt.subplot(1,2,2)
plt.imshow(p2.reshape(8,8),cmap='binary'), plt.title('Second Principal Component');

digits_2D = pca.transform(digits.data)

plt.figure(figsize=(20,8))
plt.scatter(digits_2D[:,0],digits_2D[:,1],c=digits.target,s=100,alpha=0.5,lw=0);
plt.colorbar(boundaries=range(0,11));

fives = digits.data[digits.target == 5]
eights = digits.data[digits.target == 8]

pca2 = PCA(n_components=2) # Instantiate a new PCA model with 2 components

fives_eights = np.vstack((fives,eights)) # Collect all the 5s and 8s in a single array
fives_eights_2D = pca2.fit_transform(fives_eights) # Fit and transform the 5s and 8s into 2D

fives_eights_targets = np.array([0 for _ in fives] + [1 for _ in eights]) # Hack together the array of labels for 5s and 8s

plt.figure(figsize=(20,8))
plt.scatter(fives_eights_2D[:,0],fives_eights_2D[:,1],c=fives_eights_targets,s=100,alpha=0.5,lw=0);
plt.colorbar();

X = np.matrix(digits.data) - np.mean(digits.data,axis=0)

A = X.T * X

evals, evecs = np.linalg.eig(A)

P1 = evecs[:,0]
P2 = evecs[:,1]
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.imshow(P1.reshape(8,8),cmap='binary'), plt.title('First Principal Component')
plt.subplot(1,2,2)
plt.imshow(P2.reshape(8,8),cmap='binary'), plt.title('Second Principal Component');

np.linalg.norm(P1 - p1.reshape(64,1))

np.linalg.norm(P2 - p2.reshape(64,1))

