import numpy as np
import matplotlib.pyplot as plt
import random
from Cfast_km import Cfast_km

nb_cluster = 5
nb_features = 2
nb_samples = 500
nb_cluster, nb_features,nb_samples

X = [[random.random() for i in range(nb_features)] for j in range(nb_samples)]      
mu = random.sample(X, nb_cluster)
X = np.array(X, dtype="double")
mu = np.array(mu, dtype="double")
mu

weights = np.array([1 for i in range(nb_features)], dtype="double")
labels = np.zeros(len(X), dtype="int")

Cfast_km(X , mu , labels, weights, 0)
mu, labels

cmap = { 0:'k',1:'b',2:'y',3:'g',4:'r' }

fig = plt.figure(figsize=(5,5))
plt.xlim(0,1)
plt.ylim(0,1)


for i in range(nb_cluster):
    X_extract = X[labels == i]
    X1 = np.transpose(X_extract[:,0]).tolist()
    X2 = np.transpose(X_extract[:,1]).tolist()
    plt.plot(X1, X2, cmap[i%5]+'.', alpha=0.5)


mu1 = np.transpose(mu[:,0]).tolist()
mu2 = np.transpose(mu[:,1]).tolist()
plt.plot(mu1, mu2, 'ro')

plt.show()

