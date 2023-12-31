import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')

wine_data = pd.read_csv('wine.csv')

wine_data.head()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(wine_data.values)

from sklearn.decomposition import PCA

sklearn_pca = PCA(n_components=2)

reduced_data = sklearn_pca.fit_transform(wine_data.values)

print(wine_data.shape)
print(reduced_data.shape)

plt.scatter(reduced_data[:, 0], reduced_data[:, 1], label='Wine points reduced')
plt.title("PCA sklearn")
plt.legend()
plt.show()

from pca_numpy import  *

reduced_data_numpy = PCA_numpy(wine_data.values)

plt.scatter(reduced_data_numpy[:, 0], reduced_data_numpy[:, 1], label='Wine points reduced')
plt.title("PCA numpy")
plt.legend()
plt.show()

