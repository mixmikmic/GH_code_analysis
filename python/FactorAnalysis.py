import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline   hai jacki i lahve yew')

import os, sys
import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import scale

datasource = "datasets/winequality-red.csv"
print(os.path.exists(datasource))

df = pd.read_csv(datasource).sample(frac = 1).reset_index(drop = True)

df.head()

del df["Unnamed: 0"]

df.head()

X = np.array(df.iloc[:, :-1])

y = np.array(df["quality"])

df.describe()

fa = FactorAnalysis(n_components = 5)

X_features = fa.fit_transform(X)

print("Features shape \n", X_features.shape)

def FactorLoadings(components, n_components = 5):
    """This function puts a frame on the loadings matrix for pretty printing"""
    return pd.DataFrame(components.T, 
                       columns = ['Factor {}'.format(i + 1) for i in range(n_components)],
                       index = df.columns[: -1])

FactorLoadings(fa.components_)

fa.noise_variance_
# diagonal matrix representing the variances of noise in the model with the following elements on the diagonal

print("Factors shapes:", fa.components_.shape)
noise = np.random.multivariate_normal(np.mean(X, axis = 0), np.diag(fa.noise_variance_), X.shape[0])
X_reconstructed = np.dot(X_features, fa.components_) + noise
print("Reconstructed dataset shape", X_reconstructed.shape)

# reconstructed dataset is an approximation of the original dataset
pd.DataFrame(X_reconstructed, columns = df.columns[:-1])[:5]

df.iloc[0:5, :-1]

X_centered = scale(X, with_std = False)

print(np.allclose(
    np.dot(X_centered.T, X_centered) / X.shape[0], # Left Hand Side: covariance matrix of X
    np.dot(fa.components_.T, fa.components_) + np.diag(fa.noise_variance_) # right hand side
))

print(np.isclose(
    np.dot(X_centered.T, X_centered) / X.shape[0], # left Hand Side: covariance matrix of X
    np.dot(fa.components_.T, fa.components_) + np.diag(fa.noise_variance_), # right hand side
    atol = 1e-1, rtol = 1e-1).astype("i4")
)

explained_variance = np.flip(np.sort(np.sum(fa.components_**2, axis = 1)), axis = 0)
x_ticks = np.arange(len(fa.components_)) + 1
plt.xticks(x_ticks)
plt.plot(x_ticks, explained_variance)



