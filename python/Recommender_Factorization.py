import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

M = np.array([[1, 2, 2, 0, 0],
              [3, 5, 5, 0, 0],
              [4, 4, 4, 0, 0],
              [5, 5, 5, 0, 0],
              [0, 2, 0, 4, 4],
              [0, 0, 0, 5, 5],
              [0, 1, 0, 2, 2]])

# Make interpretable
movies = ['Matrix','Alien','StarWars','Casablanca','Titanic']
users = ['Alice','Bob','Cindy','Dan','Emily','Frank','Greg']

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

M_scaled = scaler.fit_transform(M)

pca = PCA()
V_pca = pca.fit_transform(M_scaled)
pd.DataFrame(np.around(pca.components_, 2))

print np.around(V_pca, 2)
print
print np.around(np.dot(M_scaled, pca.components_.T), 2)

# Compute SVD
from numpy.linalg import svd

U, sigma, VT = svd(M_scaled)

U, sigma, VT = (np.around(x, 2) for x in (U, sigma, VT))

U = pd.DataFrame(U, index=users)
VT = pd.DataFrame(VT, columns=movies)

sigma_df = pd.DataFrame(np.diag(sigma))
sigma_df

U

VT

# Power
# singular values are square roots of eigenvalues
total_power = np.sum(sigma**2)
total_power

fraction_power = np.cumsum(sigma**2) / total_power
fraction_power

# Keep only top two concepts
U = U.iloc[:,:2]
sigma = sigma[:2]
VT = VT.iloc[:2,:]

print U
print sigma
print VT

# Check the reconstruction

np.around(U.dot(np.diag(sigma)).dot(VT))



