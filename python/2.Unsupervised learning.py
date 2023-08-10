get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 200)

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

iris = load_iris()
X = iris['data']

km = KMeans(n_clusters=3, max_iter=1000)
km.fit(X)

km.cluster_centers_

km.labels_

df = pd.DataFrame(X, columns=iris['feature_names'])
df['target'] = iris['target']
df['kmeans_lables'] = km.labels_
df.head()

new_data_point = np.array([[4.8, 4.3, 2, 0.9]])

# Rememeber variable 'km' holds the trained model of kmeans
km.transform(new_data_point)



