get_ipython().magic('pylab inline')
pylab.rcParams['figure.figsize'] = (14, 14)

from sklearn.datasets import load_digits as load_data
from sklearn.naive_bayes import GaussianNB

# This is all that's needed for scikit-plot
import matplotlib.pyplot as plt
from scikitplot import classifier_factory

# Load data
X, y = load_data(return_X_y=True)

# Regular instance using GaussianNB class
nb = GaussianNB()

# Modification of instance of Scikit-Learn
classifier_factory(nb)

# An object of Scikit-Learn using the modified version that can use a method plot_roc_curve
nb.plot_roc_curve(X, y, random_state=1)

# Display plot
plt.show()



from sklearn.ensemble import RandomForestClassifier

random_forest_clf = RandomForestClassifier(n_estimators=5, max_depth=5, random_state=1)

from scikitplot import classifier_factory

classifier_factory(random_forest_clf)

random_forest_clf.plot_confusion_matrix(X, y, normalize=True)

plt.show()



from scikitplot import plotters as skplt

rf = RandomForestClassifier()

rf = rf.fit(X, y)

preds = rf.predict(X)

skplt.plot_confusion_matrix(y_true=y, y_pred=preds)
plt.show()





from __future__ import absolute_import
import matplotlib.pyplot as plt
from scikitplot import clustering_factory
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris as load_data

X, y = load_data(return_X_y=True)

kmeans = clustering_factory(KMeans(random_state=1))

kmeans.plot_elbow_curve(X, cluster_ranges=range(1, 11))
plt.show()





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





from __future__ import absolute_import
import matplotlib.pyplot as plt
from scikitplot import classifier_factory
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer as load_data

X, y = load_data(return_X_y=True)

lr = classifier_factory(LogisticRegression())

lr.plot_ks_statistic(X, y, random_state=1)
plt.show()



from scikitplot import plotters as skplt

lr = LogisticRegression()

lr = lr.fit(X, y)

probas = lr.predict_proba(X)

skplt.plot_ks_statistic(y_true=y, y_probas=probas)
plt.show()





from __future__ import absolute_import
import matplotlib.pyplot as plt
from scikitplot import classifier_factory
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer as load_data

X, y = load_data(return_X_y=True)

rf = classifier_factory(RandomForestClassifier())

rf.plot_learning_curve(X, y)
plt.show()



from scikitplot import plotters as skplt

rf = RandomForestClassifier()

skplt.plot_learning_curve(rf, X, y)
plt.show()





from sklearn.decomposition import PCA
from sklearn.datasets import load_digits as load_data
import scikitplot.plotters as skplt
import matplotlib.pyplot as plt

X, y = load_data(return_X_y=True)

pca = PCA(random_state=1)

pca.fit(X)

skplt.plot_pca_2d_projection(pca, X, y)
plt.show()





from sklearn.decomposition import PCA
from sklearn.datasets import load_digits as load_data
import scikitplot.plotters as skplt
import matplotlib.pyplot as plt

X, y = load_data(return_X_y=True)

pca = PCA(random_state=1)

pca.fit(X)

skplt.plot_pca_component_variance(pca)
plt.show()





from __future__ import absolute_import
import matplotlib.pyplot as plt
from scikitplot import classifier_factory
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_digits as load_data

X, y = load_data(return_X_y=True)

nb = classifier_factory(GaussianNB())

nb.plot_precision_recall_curve(X, y, random_state=1)
plt.show()





from __future__ import absolute_import
import matplotlib.pyplot as plt
from scikitplot import classifier_factory
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_digits as load_data

X, y = load_data(return_X_y=True)

nb = classifier_factory(GaussianNB())

nb.plot_roc_curve(X, y, random_state=1)
plt.show()



from scikitplot import plotters as skplt

nb = GaussianNB()

nb = nb.fit(X, y)

probas = nb.predict_proba(X)

skplt.plot_roc_curve(y_true=y, y_probas=probas)
plt.show()







from __future__ import absolute_import
import matplotlib.pyplot as plt
from scikitplot import clustering_factory
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris as load_data

X, y = load_data(return_X_y=True)

kmeans = clustering_factory(KMeans(n_clusters=4, random_state=1))

kmeans.plot_silhouette(X)
plt.show()





