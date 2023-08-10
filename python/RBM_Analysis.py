import numpy as np
import h5py

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from sklearn import linear_model, datasets, metrics, preprocessing 
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.datasets import load_digits
from sklearn.utils.validation import assert_all_finite
from scipy.sparse import csc_matrix, csr_matrix, lil_matrix
from sklearn.utils.testing import (assert_almost_equal, assert_array_equal,
                                   assert_true)


from sklearn.preprocessing import Binarizer
np.seterr(all='warn')

Xdigits = load_digits().data
Xdigits -= Xdigits.min()
Xdigits /= Xdigits.max()

b = Binarizer(threshold=0.001, copy=True)
Xdigits = b.fit_transform(Xdigits)
print Xdigits.shape

from sklearn.neural_network import BernoulliRBM
X = Xdigits.copy()

def one_epoch_Brbm(X):
    rbm1 = BernoulliRBM(n_components=64, batch_size=20,
                  learning_rate=0.005, verbose=False, n_iter=1)
    rbm1.fit(X);
    return rbm1

ws1, vbs1, hbs1, scores1 = [], [], [], []
for i in range(10000):
    rbm1 = one_epoch_Brbm(X)
    ws1.append(np.linalg.norm(rbm1.components_,ord=2))
    vbs1.append(np.linalg.norm(rbm1.intercept_visible_,ord=2))
    hbs1.append(np.linalg.norm(rbm1.intercept_hidden_,ord=2))
    scores1.append(np.average(rbm1.score_samples(X)))
    if (i%100 == 0):
        print i,

def twenty_epochs_Brbm(X):
    rbm20 = BernoulliRBM(n_components=64, batch_size=20,
                  learning_rate=0.005, verbose=False, n_iter=20)
    rbm20.fit(X);
    return rbm20

ws20, vbs20, hbs20, scores20 = [], [], [], []
X = Xdigits.copy()

for i in range(10000):
    rbm20 = twenty_epochs_Brbm(X)
    ws20.append(np.linalg.norm(rbm20.components_,ord=2))
    vbs20.append(np.linalg.norm(rbm20.intercept_visible_,ord=2))
    hbs20.append(np.linalg.norm(rbm20.intercept_hidden_,ord=2))
    scores20.append(np.average(rbm20.score_samples(X)))
    if (i%100 == 0):
        print i,



plt.hist(scores1,100, color='r', alpha = 0.5);
plt.hist(scores20,100, color='g', alpha = 0.5);


plt.title("dist of initial and final free energies")

plt.hist(vbs1,100, color='red');
plt.hist(vbs20,100, color='green');

plt.title("dist of |v_bias| norms")

plt.hist(hbs1,100, color='red');
plt.hist(hbs20,100, color='green');

plt.title("dist of |h_bias| norms")

from sklearn.neighbors.kde import KernelDensity
from scipy.stats import norm
X = np.array(scores20)[:, None]
X_plot = np.linspace(-21, -29, X.shape[0])[:, None]

kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot[:, 0], np.exp(log_dens), '-')

plt.title('Fit of (final) Free Energies to Gaussian')
plt.hist(scores20,bins=100, normed=True, alpha=0.5)
plt.show()

from sklearn.neighbors.kde import KernelDensity
from scipy.stats import norm
X = np.array(scores1)[:, None]
X_plot = np.linspace(-21, -29, X.shape[0])[:, None]

kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot[:, 0], np.exp(log_dens), '-')

plt.title('Fit of (initial) Free Energies to Gaussian')
plt.hist(scores1,bins=100, normed=True, alpha=0.5, color='red')
plt.show()



