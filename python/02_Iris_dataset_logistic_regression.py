import sklearn
from sklearn import datasets

from sklearn import linear_model
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
get_ipython().magic('matplotlib inline')
from matplotlib.colors import ListedColormap

import numpy as np
from IPython.core.display import HTML
HTML("<style>.container { width:100% !important; }</style>")

iris = datasets.load_iris()

iris.feature_names

# C is the inverse of regularization parameter (smaller values specify strong regularization)
logreg = linear_model.LogisticRegression(C=1e5)

df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
prediction_errors  = []
for items in itertools.combinations(df.columns.values, r=2):
    X = df[list(items)].as_matrix()
    logreg.fit(X, iris.target)
    iris_predict = logreg.predict(X)
    diff = iris.target - iris_predict
    prediction_errors.append([items[0], items[1], diff.nonzero()[0].size])
pd.DataFrame(prediction_errors, columns=['measure 1', 'measure 2', 'incorrect predictions'])

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
iris_proj = pca.fit_transform(iris['data'])
print(iris['data'].shape)
print(iris_proj.shape)

df2 = pd.DataFrame(iris_proj, columns=['pc1', 'pc2'])
df2['species'] = pd.Categorical.from_codes(
    iris.target, categories=iris.target_names, ordered=True)

g = sns.FacetGrid(df2, hue='species', size=8)
g.map(plt.scatter, 'pc1', 'pc2')
g.set_xlabels('principal component 1')
g.set_ylabels('principal component 2')
g.add_legend()

logreg.fit(iris_proj, iris.target)
iris_predict = logreg.predict(iris_proj)
diff = iris.target - iris_predict
print(['principal component 1', 'principal component 2', diff.nonzero()[0].size])

predict_series = iris.target
predict_series[diff.astype(bool)] = 3
names = list(iris.target_names)
names.extend(['mis-predicted'])
df2['prediction'] = pd.Categorical.from_codes(
    predict_series, categories=names, ordered=True)

hue_options = {
    'marker': ['o', 'o', 'o', 'd'],
    'alpha': [0.8, 0.8, 0.8, 1]
}
sns.despine()
g = sns.FacetGrid(df2, hue='prediction', hue_kws=hue_options, size=8)
g = g.map(plt.scatter, 'pc1', 'pc2', s=50)
g.set_xlabels('principal component 1')
g.set_ylabels('principal component 2')
g.add_legend()

x_min, x_max = iris_proj[:, 0].min() - .1, iris_proj[:, 0].max() + .1
y_min, y_max = iris_proj[:, 1].min() - .1, iris_proj[:, 1].max() + .1
step = 0.04  # step size for mesh

xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                     np.arange(y_min, y_max, step))
mesh_predict = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

mesh_predict = mesh_predict.reshape(xx.shape)

g = sns.FacetGrid(df2, hue='species', size=8)
g = g.map(plt.scatter, 'pc1', 'pc2', s=50)
plt.gca().pcolormesh(xx, yy, mesh_predict,
                     cmap=ListedColormap(sns.color_palette()[:3]), alpha=0.2)
g.set_xlabels('principal component 1')
g.set_ylabels('principal component 2')
g.add_legend()



