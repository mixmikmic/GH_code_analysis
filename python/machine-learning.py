from dask.distributed import Client, progress
client = Client(processes=False, threads_per_worker=4, n_workers=1, memory_limit='2GB')
client

import dask_ml.joblib  # register the distriubted backend
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pandas as pd

X, y = make_classification(n_samples=1000, random_state=0)
X[:5]

param_grid = {"C": [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
              "kernel": ['rbf', 'poly', 'sigmoid'],
              "shrinking": [True, False]}

grid_search = GridSearchCV(SVC(gamma='auto', random_state=0, probability=True),
                           param_grid=param_grid,
                           return_train_score=False,
                           iid=True,
                           n_jobs=-1)

from sklearn.externals import joblib

with joblib.parallel_backend('dask', scatter=[X, y]):
    grid_search.fit(X, y)

pd.DataFrame(grid_search.cv_results_).head()

grid_search.predict(X)[:5]

grid_search.score(X, y)

get_ipython().run_line_magic('matplotlib', 'inline')

import dask_ml.datasets
import dask_ml.cluster
import matplotlib.pyplot as plt

X, y = dask_ml.datasets.make_blobs(n_samples=10000000,
                                   chunks=1000000,
                                   random_state=0,
                                   centers=3)
X = X.persist()
X

km = dask_ml.cluster.KMeans(n_clusters=3, init_max_iter=2, oversampling_factor=10)
km.fit(X)

fig, ax = plt.subplots()
ax.scatter(X[::10000, 0], X[::10000, 1], marker='.', c=km.labels_[::10000],
           cmap='viridis', alpha=0.25);

