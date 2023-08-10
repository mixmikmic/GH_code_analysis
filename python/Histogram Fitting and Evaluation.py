get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
import matplotlib.pyplot as plt

import numpy as np
import theano
from scipy.stats import chi2
from itertools import product

import os, sys, inspect
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder + '/../systematics/carl')

import carl

np.random.seed(314)

from carl.distributions import Normal
from carl.distributions import Mixture

components_0 = [
    Normal(mu=-2.0, sigma=0.75),   # c0
    Normal(mu=0.0, sigma=2.0),     # c1
    Normal(mu=1.0, sigma=0.5)      # c2 (bump)
]

components_1 = [
    Normal(mu=-3.0, sigma=0.55),   # c0
    Normal(mu=1.0, sigma=1.0),     # c1
    Normal(mu=2.0, sigma=0.5)      # c2 (bump)
]

bump_coefficient = 0.05
g = theano.shared(bump_coefficient) 
#p0 = Mixture(components=components_0, weights=[0.5 - g / 2., 0.5 - g / 2., g])
#p1 = Mixture(components=components_1, weights=[0.5 - g / 2., 0.5 - g / 2., g])
p0 = Normal(mu=-2.0, sigma=0.75)
p1 = Normal(mu=-1., sigma=0.5)

X_true = p0.rvs(5000, random_state=777)

reals = np.linspace(-5, 5, num=1000)
plt.plot(reals, p0.pdf(reals.reshape(-1, 1)), label=r"$p(x|\gamma=0.05)$", color="b")
plt.plot(reals, p1.pdf(reals.reshape(-1, 1)), label=r"$p(x|\gamma=0)$", color="r")
plt.hist(X_true[:, 0], bins=100, normed=True, label="data", alpha=0.2, color="b")
plt.xlim(-5, 5)
plt.legend(loc="best", prop={'size': 8})
#plt.savefig("fig1a.pdf")
plt.show()

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPRegressor
from carl.ratios import ClassifierRatio
from carl.ratios import DecomposedRatio
from carl.learning import CalibratedClassifierCV
from carl.learning import as_classifier

n_samples = 200000
clf = as_classifier(MLPRegressor(tol=1e-05, activation="logistic", 
                   hidden_layer_sizes=(10, 10), learning_rate_init=1e-07, 
                   learning_rate="constant", algorithm="l-bfgs", random_state=1, 
                   max_iter=75))

p0_data = p0.rvs(n_samples // 2, random_state=1234)
p1_data = p1.rvs(n_samples // 2, random_state=1234)

X = np.vstack((p0_data, p1_data))
y = np.zeros(n_samples, dtype=np.int)
y[n_samples//2:] = 1

clf.fit(X=X, y=y)

# Calibration + Direct approximation 
#cv = StratifiedShuffleSplit(n_iter=1, test_size=0.5, random_state=1)
cc_direct = ClassifierRatio(
    base_estimator=CalibratedClassifierCV(clf, cv='prefit', bins=20, var_width=False), 
    random_state=0)

cc_direct.fit(X=X, y=y)

import carl
reals2 = np.linspace(0, 1, 100)
x = np.linspace(0.,1,300)
cal_num, cal_den = cc_direct.classifier_.calibrators_[0].calibrators
print(cal_num.amise(biased=False))
print(cal_num.amise(biased=True))
plt.plot(x, cal_num.pdf(x.reshape(-1, 1)), 
         label="p(s_num=c,den=c, x", c='b')
plt.plot(x, cal_den.pdf(x.reshape(-1, 1)), 
        label="p(s_num=c,den=c), x~c", c='r')
plt.legend()
plt.show()

import carl
amises_biased_num = []
amises_unbiased_num = []
amises_biased_den = []
amises_unbiased_den = []
h_num, h_den = [],[]
#for n_bins in range(30,50,1):
for h in np.linspace(0.1,0.9,50):
    n_bins = np.ceil((X.max()-X.min())/h)
    print(str(n_bins) + ' '),
    cc_direct = ClassifierRatio(
        base_estimator=CalibratedClassifierCV(clf, cv='prefit', bins=n_bins, var_width=False), 
        random_state=0)
    cc_direct.fit(X=X, y=y)

    cal_num, cal_den = cc_direct.classifier_.calibrators_[0].calibrators
    amises_biased_num.append(cal_num.amise(biased=True))
    amises_unbiased_num.append(cal_num.amise(biased=False))
    amises_biased_den.append(cal_den.amise(biased=True))
    amises_unbiased_den.append(cal_den.amise(biased=False))
    h_num.append(cal_num.edges_[0][2]-cal_num.edges_[0][1])
    h_den.append(cal_den.edges_[0][2]-cal_den.edges_[0][1])
    #print(cal_num.oversmoothed_bins(X[:n_samples//2]))

# Using numpu methods to estimate number of bins
methods = ['auto', 'fd', 'doane', 'scott', 'rice', 'sturges', 'sqrt']
opt_bins = []
opt_hs = []
for m in methods:
    cc_direct = ClassifierRatio(
        base_estimator=CalibratedClassifierCV(clf, cv='prefit', bins=m, var_width=False), 
        random_state=0)
    cc_direct.fit(X=X, y=y)
    cal_num, cal_den = cc_direct.classifier_.calibrators_[0].calibrators
    opt_hs.append(cal_num.edges_[0][2]-cal_num.edges_[0][1])
    opt_bins.append(len(cal_num.edges_[0]) - 3)

#print(h_num)
#xs = range(10,80,1)
hs = np.linspace(0.1,0.9,50)
plt.subplot(211)
plt.plot(hs, amises_biased_num)
plt.subplot(212)         
plt.plot(hs, amises_unbiased_num)
opt_h_bcv = hs[np.argmin(amises_biased_num)]
opt_bin_bcv = np.ceil((X.max()-X.min())/opt_h_bcv)
opt_h_ucv = hs[np.argmin(amises_unbiased_num)]
opt_bin_ucv = np.ceil((X.max()-X.min())/opt_h_ucv)
print('Width upper bound: {0}'.format(cal_num.oversmoothed_bins(X[:n_samples//2])))
print('The optimum bin width is (BCV) : {0}'.format(opt_h_bcv))
print('The optimum number of bins is (BCV) : {0}'.format(opt_bin_bcv))
print('The optimum bin width is (UCV) : {0}'.format(opt_h_ucv))
print('The optimum number of bins is (UCV) : {0}'.format(opt_bin_ucv))
print('Numpy methods:')
print(methods)
print('Numpy optimal widths:')
print(opt_hs)
print('Numpy optimal bins:')
print(opt_bins)

cc_direct = ClassifierRatio(
    base_estimator=CalibratedClassifierCV(clf, cv='prefit', bins=47, var_width=False), 
    random_state=0)
cc_direct.fit(X=X, y=y)
cal_num, cal_den = cc_direct.classifier_.calibrators_[0].calibrators
reals2 = np.linspace(0, 1, 100)
x = np.linspace(0.,1,300)
cal_num, cal_den = cc_direct.classifier_.calibrators_[0].calibrators
print(cal_num.amise(biased=False))
print(cal_num.amise(biased=True))
plt.plot(x, cal_num.pdf(x.reshape(-1, 1)), 
         label="p(s_num=c,den=c, x", c='b')
plt.plot(x, cal_den.pdf(x.reshape(-1, 1)), 
        label="p(s_num=c,den=c), x~c", c='r')
plt.legend()
plt.show()



