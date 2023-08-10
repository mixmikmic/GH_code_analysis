get_ipython().magic('matplotlib inline')

import pandas as pd
import numpy as np
from sklearn import cluster, metrics
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot') 
from sklearn import datasets
import seaborn as sns

# A:
iris_data = datasets.load_iris()

print iris_data.DESCR

iris_data.target

iris_data.target_names

# A:
iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
target_df = pd.DataFrame(iris_data.target, columns=['species'])
df = pd.concat([iris_df, target_df], axis=1)

sns.pairplot(df, hue='species')

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, silhouette_samples, silhouette_score

# Run k-means
k = 3
kmeans = KMeans(n_clusters=k)

# A:
clusters = kmeans.fit_predict(iris_df)
centroids = kmeans.cluster_centers_

# A:

# A:

# A:

# A:

# A:

# A:

# A:

