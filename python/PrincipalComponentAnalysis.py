import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import os, sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

datasource = "datasets/winequality-red.csv"
print(os.path.exists(datasource))

df = pd.read_csv(datasource).sample(frac = 1).reset_index(drop = True)
del df["Unnamed: 0"]

df.head()

df.describe()

X = np.array(df.iloc[:, :-1]) # everything except for quality

y = np.array(df["quality"]) # just the quality column

pca = PCA(n_components = 5)

pca.fit(X)

print(pca.explained_variance_ratio_)

X_features = pca.transform(X)

print(X_features)

print("Features Shape:\n", X_features.shape)

print("Principal components shape:", pca.components_.shape)

X_synthesized = np.dot(X_features, pca.components_)

X_reconstructed = X_synthesized + np.mean(X, axis = 0)[np.newaxis, ...]

print("Reconstructed dataset shape:", X_reconstructed.shape)

# reconstructed dataset, an approximation of original dataset
pd.DataFrame(X_reconstructed, columns = df.columns[:-1])[0:5]

df.iloc[0:5, :-1]

error = X - X_reconstructed

np.mean((error**2))

x_ticks = np.arange(len(pca.components_)) + 1
plt.xticks(x_ticks) # this enforces integers on the x axis
plt.plot(x_ticks, pca.explained_variance_)

plt.xticks(x_ticks)
plt.plot(x_ticks, pca.explained_variance_ratio_)
print("Total Explained Variance Ratio\n", np.sum(pca.explained_variance_ratio_))



