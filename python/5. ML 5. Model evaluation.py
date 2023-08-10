get_ipython().run_cell_magic('bash', '', 'git clone https://github.com/reiinakano/scikit-plot.git\ncd ')

get_ipython().run_cell_magic('bash', '', 'cd scikit-plot\ndir\npython setup.py install --user')

get_ipython().magic('matplotlib inline')
from sklearn.datasets import load_digits as load_data
from sklearn.naive_bayes import GaussianNB

# This is all that's needed for scikit-plot
import matplotlib.pyplot as plt
from scikitplot import classifier_factory

X, y = load_data(return_X_y=True)
nb = GaussianNB()
classifier_factory(nb)
nb.plot_roc_curve(X, y, random_state=1)
plt.show()

"""An example showing the plot_silhouette method used by a scikit-learn clusterer"""
from __future__ import absolute_import
import matplotlib.pyplot as plt
from scikitplot import clustering_factory
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris as load_data


X, y = load_data(return_X_y=True)
kmeans = clustering_factory(KMeans(random_state=1))
kmeans.plot_elbow_curve(X, cluster_ranges=range(1, 11))
plt.show()

"""An example showing the plot_feature_importances method used by a scikit-learn classifier"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris as load_data
import matplotlib.pyplot as plt
from scikitplot import classifier_factory

X, y = load_data(return_X_y=True)
rf = classifier_factory(RandomForestClassifier(random_state=1))
rf.fit(X, y)
rf.plot_feature_importances(feature_names=['petal length', 'petal width',
                                           'sepal length', 'sepal width'])
plt.show()

# Using the more flexible functions API
from scikitplot import plotters as skplt
rf = RandomForestClassifier()
rf = rf.fit(X, y)
skplt.plot_feature_importances(rf, feature_names=['petal length', 'petal width',
                                                  'sepal length', 'sepal width'])
plt.show()

"""An example showing the plot_ks_statistic method used by a scikit-learn classifier"""
from __future__ import absolute_import
import matplotlib.pyplot as plt
from scikitplot import classifier_factory
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer as load_data


X, y = load_data(return_X_y=True)
lr = classifier_factory(LogisticRegression())
lr.plot_ks_statistic(X, y, random_state=1)
plt.show()

# Using the more flexible functions API
from scikitplot import plotters as skplt
lr = LogisticRegression()
lr = lr.fit(X, y)
probas = lr.predict_proba(X)
skplt.plot_ks_statistic(y_true=y, y_probas=probas)
plt.show()

"""An example showing the plot_learning_curve method used by a scikit-learn classifier"""
from __future__ import absolute_import
import matplotlib.pyplot as plt
from scikitplot import classifier_factory
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer as load_data


X, y = load_data(return_X_y=True)
rf = classifier_factory(RandomForestClassifier())
rf.plot_learning_curve(X, y)
plt.show()

# Using the more flexible functions API
from scikitplot import plotters as skplt
rf = RandomForestClassifier()
skplt.plot_learning_curve(rf, X, y)
plt.show()

"""An example showing the plot_pca_2d_projection method used by a scikit-learn PCA object"""
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits as load_data
import scikitplot.plotters as skplt
import matplotlib.pyplot as plt

X, y = load_data(return_X_y=True)
pca = PCA(random_state=1)
pca.fit(X)
skplt.plot_pca_2d_projection(pca, X, y)
plt.show()

"""An example showing the plot_pca_component_variance method used by a scikit-learn PCA object"""
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits as load_data
import scikitplot.plotters as skplt
import matplotlib.pyplot as plt


X, y = load_data(return_X_y=True)
pca = PCA(random_state=1)
pca.fit(X)
skplt.plot_pca_component_variance(pca)
plt.show()

"""An example showing the plot_precision_recall method used by a scikit-learn classifier"""
from __future__ import absolute_import
import matplotlib.pyplot as plt
from scikitplot import classifier_factory
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_digits as load_data


X, y = load_data(return_X_y=True)
nb = classifier_factory(GaussianNB())
nb.plot_precision_recall_curve(X, y, random_state=1)
plt.show()

# Using the more flexible functions API
from scikitplot import plotters as skplt
nb = GaussianNB()
nb = nb.fit(X, y)
probas = nb.predict_proba(X)
skplt.plot_precision_recall_curve(y_true=y, y_probas=probas)
plt.show()

"""An example showing the plot_roc_curve method used by a scikit-learn classifier"""
from __future__ import absolute_import
import matplotlib.pyplot as plt
from scikitplot import classifier_factory
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_digits as load_data


X, y = load_data(return_X_y=True)
nb = classifier_factory(GaussianNB())
nb.plot_roc_curve(X, y, random_state=1)
plt.show()

# Using the more flexible functions API
from scikitplot import plotters as skplt
nb = GaussianNB()
nb = nb.fit(X, y)
probas = nb.predict_proba(X)
skplt.plot_roc_curve(y_true=y, y_probas=probas)
plt.show()

"""An example showing the plot_silhouette method used by a scikit-learn clusterer"""
from __future__ import absolute_import
import matplotlib.pyplot as plt
from scikitplot import clustering_factory
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris as load_data


X, y = load_data(return_X_y=True)
kmeans = clustering_factory(KMeans(n_clusters=4, random_state=1))
kmeans.plot_silhouette(X)
plt.show()

