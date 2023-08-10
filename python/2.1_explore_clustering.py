get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# use seaborn plotting defaults
import seaborn as sns; sns.set()

from sklearn.datasets.samples_generator import make_blobs
X, y = make_blobs(n_samples=300, centers=4,
                  random_state=0, cluster_std=0.60)
plt.scatter(X[:, 0], X[:, 1], s=50);

from sklearn.cluster import KMeans
est = KMeans(4)  # 4 clusters
est.fit(X)
y_kmeans = est.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='rainbow');

from networkplots import plot_kmeans_interactive
plot_kmeans_interactive();

from bokeh.io import output_notebook

# This line is required for the plots to appear in the notebooks
output_notebook()

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

import networkplots

networkplots.explore_phenograph()

