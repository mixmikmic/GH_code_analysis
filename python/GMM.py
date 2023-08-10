# This tells matplotlib not to try opening a new window for each plot.
get_ipython().magic('matplotlib inline')

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.mixture import GMM
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA

from sklearn.cluster import AgglomerativeClustering 
from sklearn.cluster import KMeans 

# data generation
np.random.seed(1)

covar = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

dat_threedee = np.vstack((np.random.multivariate_normal([1, 1, 1], covar, 250), 
                          np.random.multivariate_normal([1, 5, 1], covar, 250),
                          np.random.multivariate_normal([3, 2, 1], covar, 250)))

dat_twodee = np.vstack((np.random.multivariate_normal([0, 0], [[10, 0], [0, 1]], 100),
                        np.random.multivariate_normal([0, 5], [[1, 2], [2, 1]], 100),
                        np.random.multivariate_normal([5, 5], [[5, 3], [3, 2]], 100)))

dat_twodee_simple = np.vstack((np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 100),
                               np.random.multivariate_normal([0, 5], [[1, 0], [0, 1]], 100),
                               np.random.multivariate_normal([5, 5], [[2, 0], [0, 2]], 100)))

dat_twodee_headache = np.vstack((np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 100),
                                 np.random.multivariate_normal([0, 1], [[1, 0], [0, 1]], 100),
                                 np.random.multivariate_normal([1, 1], [[2, 0], [0, 2]], 100)))

comps = 2
gm_mod = GMM(n_components = comps)

gm_mod.fit(dat_twodee)
y_hat = gm_mod.predict(dat_twodee)

print "The mean(s) estimated with this", comps, "component GMM: \n", gm_mod.means_

print "\n\nThe covariances [SEE THE DOCS] estimated with this", comps, "component GMM: \n", gm_mod.covars_

cm_bright = ListedColormap(['#FF0000', '#0000FF'])
p = plt.subplot(1, 1, 1)
p.scatter(dat_twodee[:, 0], dat_twodee[:, 1], c=y_hat, cmap=cm_bright)
plt.title("Mysterious Rabbit Data")

def proc(dat):
    km = KMeans(n_clusters=2)
    km.fit(dat)

    gm_mod = GMM(n_components = 2)
    gm_mod.fit(dat)

    ag = AgglomerativeClustering(n_clusters = 2, linkage = "complete")

    cm_bright = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#FFFFFF'])

    plt.figure(figsize=(12, 4))

    p = plt.subplot(1, 3, 1)
    p.scatter(dat[:, 0], dat[:, 1], c=km.predict(dat), cmap=cm_bright)
    plt.title('KMeans') 

    p = plt.subplot(1, 3, 2)
    p.scatter(dat[:, 0], dat[:, 1], c=gm_mod.predict(dat), cmap=cm_bright)
    plt.title('GMM') 

    p = plt.subplot(1, 3, 3)
    p.scatter(dat[:, 0], dat[:, 1], c=ag.fit_predict(dat), cmap=cm_bright)
    plt.title('Agglomerative (complete)') 
    
proc(dat_twodee)



