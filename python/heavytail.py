from __future__ import print_function

import numpy as np
from sklearn.linear_model import Ridge
from flexible_linear import FlexibleLinearRegression

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
get_ipython().magic('matplotlib inline')

np.random.seed(1)

N = 500
A = 50
B = 3
x = np.linspace(0, 100, N, dtype=float)
X = x.reshape(-1, 1)  # scikit-learn wants 2d arrays
y = A + B * x
plt.plot(x, y, '-')

y_gauss = y + 20*np.random.randn(N)
plt.plot(x, y, '-', x, y_gauss, '-')
plt.legend(['True', 'Noisy'], loc='upper left')

y_cauchy = y + 20*np.random.standard_cauchy(N)
plt.plot(x, y, '-', x, y_cauchy, '.')
plt.legend(['True', 'Noisy'], loc='lower right')

clf = Ridge()
clf.fit(X, y_gauss)
pred = clf.predict(X)
plt.plot(x, y, '-', x, pred, '-')
plt.legend(['True', 'Recovered'], loc='upper left')
print("     True: %.3f + %.3f * x" % (A, B))
print("Recovered: %.3f + %.3f * x" % (clf.intercept_, clf.coef_[0]))

clf = Ridge()
clf.fit(X, y_cauchy)
pred = clf.predict(X)
plt.plot(x, y, '-', x, pred, '-')
plt.legend(['True', 'Recovered'], loc='upper left')
print("     True: %.3f + %.3f * x" % (A, B))
print("Recovered: %.3f + %.3f * x" % (clf.intercept_, clf.coef_[0]))

clf = FlexibleLinearRegression(cost_func='l2', C=0.0)
clf.fit(X, y_cauchy)
pred = clf.predict(X)
plt.plot(x, y, '-', x, pred, '-')
plt.legend(['True', 'Recovered'], loc='upper left')
print("     True: %.3f + %.3f * x" % (A, B))
print("Recovered: %.3f + %.3f * x" % (clf.coef_[0], clf.coef_[1]))

clf = FlexibleLinearRegression(cost_func='l1', C=0.0)
clf.fit(X, y_cauchy)
pred = clf.predict(X)
plt.plot(x, y, '-', x, pred, '-')
plt.legend(['True', 'Recovered'], loc='upper left')
print("     True: %.3f + %.3f * x" % (A, B))
print("Recovered: %.3f + %.3f * x" % (clf.coef_[0], clf.coef_[1]))

clf = FlexibleLinearRegression(cost_func='japanese', C=0.0, cost_opts={'eta': 10.0})
clf.fit(X, y_cauchy)
pred = clf.predict(X)
plt.plot(x, y, '-', x, pred, '-')
plt.legend(['True', 'Recovered'], loc='upper left')
print("     True: %.3f + %.3f * x" % (A, B))
print("Recovered: %.3f + %.3f * x" % (clf.coef_[0], clf.coef_[1]))

