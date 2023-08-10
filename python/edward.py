get_ipython().magic('matplotlib inline')
from edward.models import Normal

import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from sklearn.cross_validation import train_test_split
from sklearn import linear_model, metrics

# Generate Data
N = 200 # Number of datapoints - 100 train, 100 test
M = N // 2
K = 10 # Number of features
X = np.random.randn(N, K)
y = np.dot(X, np.random.randn(K)) + np.random.normal(0, 0.5, size=(N))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# Apply scikit learn linear model
lr, ridge, lasso = linear_model.LinearRegression(), linear_model.Ridge(alpha=0.1), linear_model.Lasso(alpha=0.1)
lr.fit(X_train, y_train)
ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)
y_lr, y_ridge, y_lasso = lr.predict(X_test), ridge.predict(X_test), lasso.predict(X_test)
print('LR MSE: {}'.format(metrics.mean_squared_error(y_test, y_lr)))
print('Ridge MSE: {}'.format(metrics.mean_squared_error(y_test, y_ridge)))
print('Lasso MSE: {}'.format(metrics.mean_squared_error(y_test, y_lasso)))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='test')
plt.plot(y_lr, label='lr')
plt.plot(y_ridge, label='ridge')
plt.plot(y_lasso, label='lasso')
plt.legend()

X = tf.placeholder(tf.float32, [M, K])
w = Normal(mu=tf.zeros(K), sigma=tf.ones(K))
b = Normal(mu=tf.zeros(1), sigma=tf.ones(1))
y = Normal(mu=ed.dot(X, w) + b, sigma=tf.ones(M))

qw = Normal(mu=tf.Variable(tf.random_normal([K])),
            sigma=tf.nn.softplus(tf.Variable(tf.random_normal([K]))))
qb = Normal(mu=tf.Variable(tf.random_normal([1])),
            sigma=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))
inference = ed.KLqp({w: qw, b: qb}, data={X: X_train, y: y_train})
inference.run(n_samples=5, n_iter=250)

# CRITICISM
y_post = ed.copy(y, {w: qw.mean(), b: qb.mean()})
# This is equivalent to
# y_post = Normal(mu=ed.dot(X, qw.mean()) + qb.mean(), sigma=tf.ones(N))

print("Mean squared error on test data:")
print(ed.evaluate('mean_squared_error', data={X: X_test, y_post: y_test}))



