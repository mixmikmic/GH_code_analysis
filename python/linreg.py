get_ipython().magic('matplotlib inline')
import pymc3 as pm
import theano.tensor as T
import theano
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_moons, make_regression

X, Y = make_regression(n_samples=100, n_features=3, n_informative=1, n_targets=1, noise=5)
X = scale(X)
Y = Y.reshape(Y.shape[0], -1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.9)

fig, ax = plt.subplots()
ax.scatter(X[:, 0], Y, color='r', label='feature 0', alpha=0.5)
ax.scatter(X[:, 1], Y, color='b', label='feature 1', alpha=0.5)
ax.scatter(X[:, 2], Y, color='orange', label='feature 2', alpha=0.5)
sns.despine()
ax.legend(loc='best')
plt.savefig('scatter.png')

linreg_input = theano.shared(X_train)
linreg_output = theano.shared(Y_train)

with pm.Model() as linreg:
    weights = pm.Normal('weights', mu=0, sd=1, shape=(X.shape[1], Y.shape[1]))
    # weights = pm.Uniform('weights', 0, 1, shape=(X.shape[1], Y.shape[1]))
    biases = pm.Normal('biases', mu=0, sd=1, shape=(Y.shape[1]))
    
    output = pm.Normal('Y', T.dot(linreg_input, weights) + biases , observed=linreg_output)

get_ipython().run_cell_magic('time', '', '\nwith linreg:\n    # Run ADVI which returns posterior means, standard deviations, and the evidence lower bound (ELBO)\n    v_params = pm.variational.advi(n=500000)')

plt.plot(v_params.elbo_vals)
plt.savefig('elbo.png')

with linreg:
    trace = pm.variational.sample_vp(v_params, draws=5000)
    
pm.traceplot(trace)
plt.savefig('trace.png')

linreg_input.set_value(X_test)
linreg_output.set_value(Y_test)

ppc = pm.sample_ppc(trace, model=linreg, samples=10)

ycol = 0
ys = Y_test[:, ycol]
preds = ppc['Y'].mean(axis=0)[:, ycol]
yerr = ppc['Y'].std(axis=0)[:, ycol] * 3  # 3 standard deviations covers about 95% of the distribution
plt.errorbar(ys, preds, yerr=yerr, ls='none', marker='o')
plt.savefig('predictions.png')

pm.summary(trace)

fig, ax = plt.subplots()
means = v_params.means['weights']
stds = v_params.stds['weights']
errors = stds * 3
barwidth = 0.4
plt.bar(np.arange(len(means))-barwidth/2, means[:, 0], color='r', width=barwidth, alpha=0.3)
plt.errorbar(np.arange(len(means)), means[:, 0], color='r', yerr=errors[:, 0]*3, ls='none')
plt.xlim(-1, 3)
plt.savefig('weights.png')



