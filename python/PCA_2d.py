import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

##Dummy data
x1 = np.array([1,2,3,4,5,6])
x2 = np.array([7.5, 11, 16, 18, 20, 26])

plt.scatter(x1,x2)
plt.show()

## Combining two array to give a 2d array
X = np.c_[x1, x2]
X

pca = PCA()
X_transformed = pca.fit_transform(X)
X_transformed

##Unit vector of the two transformed directions
pca.components_

## inverse back to original data
pca.inverse_transform(X_transformed)

pca = PCA(n_components=1)       ## n_components tell number of features/dimensions to keep
X_transformed = pca.fit_transform(X)
X_transformed

## Lost some data (one dimension) so data is close to original one but not exact
## But not lost so much information.
## Thus this direction actually stored maximum information even when we are reducing the dimensions
X_approx = pca.inverse_transform(X_transformed)
X_approx

plt.scatter(X_approx[:,0], X_approx[:,1])
plt.show()
##Shows principal component direction ie. the direction which stores maximum information and it lies on a line.

