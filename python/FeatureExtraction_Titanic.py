import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import os, sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import scale
from scipy.stats import pearsonr

datasource = "datasets/titanic.csv"
print(os.path.exists(datasource))

df = pd.read_csv(datasource).sample(frac = 1).reset_index(drop = True)

df.head()

X = np.array(df.iloc[:, :-1])

y = np.array(df["survived"])

print(X.shape)

print(y.shape)

pca = PCA(n_components = 5)

pca.fit(X)

fa = FactorAnalysis(n_components = 5)

fa.fit(X)

print("PCA:\n", pca.explained_variance_ratio_)

def FA_explained_variance_ratio(fa):
    fa.explained_variance_ = np.flip(np.sort(np.sum(fa.components_**2, axis = 1)), axis = 0)
    total_variance = np.sum(fa.explained_variance_) + np.sum(fa.noise_variance_)
    fa.explained_variance_ratio_ = fa.explained_variance_ / total_variance

FA_explained_variance_ratio(fa)
    
print("FA:\n", fa.explained_variance_ratio_)

X_PCA = pca.transform(X)

corr = np.array([pearsonr(X_PCA[:,i], y)[0] for i in range(X_PCA.shape[1])])
print("PCA Correlation Coefficient:\n", corr)

X_FA = fa.transform(X)

fa_corr = [pearsonr(X_FA[:,i], y)[0] for i in range(X_FA.shape[1])]
print("FA Correlation Coefficient:\n", fa_corr)

x_ticks = np.arange(len(pca.components_)) + 1
plt.xticks(x_ticks) 
plt.plot(x_ticks, pca.explained_variance_)

x_ticks = np.arange(len(fa.components_)) + 1
plt.xticks(x_ticks)

explained_variance = np.flip(np.sort(np.sum(fa.components_**2, axis = 1)), axis = 0)
plt.plot(x_ticks, explained_variance)

x_ticks = np.arange(len(pca.components_)) + 1
plt.xticks(x_ticks)
plt.plot(x_ticks, np.log(explained_variance), c = "b") # FA
plt.plot(x_ticks, np.log(pca.explained_variance_), c = "r") # PCA
plt.show()

