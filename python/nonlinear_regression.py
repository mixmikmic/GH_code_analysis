get_ipython().magic('matplotlib inline')
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, Normalize
from tempfile import NamedTemporaryFile
import seaborn as sns
import numpy as np
import six
import tensorflow as tf

plt.style.use('seaborn-talk')
sns.set_context("talk", font_scale=1.4)
sess = ed.get_session()

sns.palplot(sns.color_palette())

# this can be done only before using Edward
ed.set_seed(42)

from edward.models import Normal

# DATA
N = 1000  # number of training data points
Np = 100  # number of test data points
D = 1  # number of features

weights_true = sess.run(Normal(loc=tf.ones(D) * 1.25,
                               scale=tf.ones(D) * 0.1))  # unknown true weights
intercept_true = sess.run(Normal(loc=tf.zeros(1),
                                 scale=tf.ones(1)))  # unknown true intercept
noise_true = 0.1  # unknown true amount of noise

def target_function(x):
    return tf.sin(tf.square(ed.dot(x, weights_true))) + intercept_true

def build_dataset(N):
    x = Normal(loc=tf.zeros([N, D]), scale=tf.ones([N, D]))
    y = Normal(loc=target_function(x), scale=noise_true)
    return sess.run([x, y])

x_train, y_train = build_dataset(N)
x_test, y_test = build_dataset(Np)

plt.scatter(x_train, y_train, s=20.0);
plt.scatter(x_test, y_test, s=20.0,
            color=sns.color_palette().as_hex()[2]);

# MODEL A
def neural_network_with_2_layers(x, W_0, W_1, b_0, b_1):
    h = tf.nn.tanh(tf.matmul(x, W_0) + b_0)
    h = tf.matmul(h, W_1) + b_1
    return tf.reshape(h, [-1])

dim = 10  # layer dimensions

W_0 = Normal(loc=tf.zeros([D, dim]),
             scale=tf.ones([D, dim]))
W_1 = Normal(loc=tf.zeros([dim, 1]),
             scale=tf.ones([dim, 1]))
b_0 = Normal(loc=tf.zeros(dim),
             scale=tf.ones(dim))
b_1 = Normal(loc=tf.zeros(1),
             scale=tf.ones(1))

x = tf.placeholder(tf.float32, [N, D])
y = Normal(loc=neural_network_with_2_layers(x, W_0, W_1, b_0, b_1),
           scale=tf.ones(N) * 0.1)  # constant noise

# BACKWARD MODEL A
q_W_0 = Normal(loc=tf.Variable(tf.random_normal([D, dim])),
               scale=tf.nn.softplus(tf.Variable(tf.random_normal([D, dim]))))
q_W_1 = Normal(loc=tf.Variable(tf.random_normal([dim, 1])),
               scale=tf.nn.softplus(tf.Variable(tf.random_normal([dim, 1]))))
q_b_0 = Normal(loc=tf.Variable(tf.random_normal([dim])),
               scale=tf.nn.softplus(tf.Variable(tf.random_normal([dim]))))
q_b_1 = Normal(loc=tf.Variable(tf.random_normal([1])),
               scale=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))

# INFERENCE A
# this will take a couple of minutes
inference = ed.KLqp(latent_vars={W_0: q_W_0, b_0: q_b_0,
                                 W_1: q_W_1, b_1: q_b_1},
                    data={x: x_train, y: y_train})
inference.run(n_samples=10, n_iter=25000)

# CRITICISM A
plt.scatter(x_train, y_train, s=20.0);  # blue
plt.scatter(x_test, y_test, s=20.0,  # red
            color=sns.color_palette().as_hex()[2]);

xp = tf.placeholder(tf.float32, [1000, D])
[plt.plot(np.linspace(-4.0, 4.0, 1000),
          sess.run(neural_network_with_2_layers(xp,
                                                q_W_0, q_W_1,
                                                q_b_0, q_b_1),
                   {xp: np.linspace(-4.0, 4.0, 1000)[:, np.newaxis]}),
          color='black', alpha=0.1)
 for _ in range(50)];

plt.plot(np.linspace(-4.0, 4.0, 1000),
         sess.run(target_function(xp),  # blue
                  {xp: np.linspace(-4.0, 4.0, 1000)[:, np.newaxis]}));

xp = tf.placeholder(tf.float32, [Np, D])
y_post = Normal(loc=neural_network_with_2_layers(xp,
                                                 q_W_0, q_W_1,
                                                 q_b_0, q_b_1),
                scale=tf.ones(Np) * 0.1)

print("Mean squared error on test data:")
print(ed.evaluate('mean_squared_error', data={xp: x_test, y_post: y_test}))

print("Mean absolute error on test data:")
print(ed.evaluate('mean_absolute_error', data={xp: x_test, y_post: y_test}))

# MODEL B
def neural_network_with_3_layers(x, W_0, W_1, W_2, b_0, b_1, b_2):
    h = tf.tanh(tf.matmul(x, W_0) + b_0)
    h = tf.tanh(tf.matmul(h, W_1) + b_1)
    h = tf.matmul(h, W_2) + b_2
    return tf.reshape(h, [-1])

dim = 10  # layer dimensions

W_0 = Normal(loc=tf.zeros([D, dim]),
             scale=tf.ones([D, dim]))
W_1 = Normal(loc=tf.zeros([dim, dim]),
             scale=tf.ones([dim, dim]))
W_2 = Normal(loc=tf.zeros([dim, 1]),
             scale=tf.ones([dim, 1]))
b_0 = Normal(loc=tf.zeros(dim),
             scale=tf.ones(dim))
b_1 = Normal(loc=tf.zeros(dim),
             scale=tf.ones(dim))
b_2 = Normal(loc=tf.zeros(1),
             scale=tf.ones(1))

x = tf.placeholder(tf.float32, [N, D])
y = Normal(loc=neural_network_with_3_layers(x, W_0, W_1, W_2, b_0, b_1, b_2),
           scale=tf.ones(N) * 0.1)  # constant noise

# BACKWARD MODEL B
q_W_0 = Normal(loc=tf.Variable(tf.random_normal([D, dim])),
               scale=tf.nn.softplus(tf.Variable(tf.random_normal([D, dim]))))
q_W_1 = Normal(loc=tf.Variable(tf.random_normal([dim, dim])),
               scale=tf.nn.softplus(tf.Variable(tf.random_normal([dim, dim]))))
q_W_2 = Normal(loc=tf.Variable(tf.random_normal([dim, 1])),
               scale=tf.nn.softplus(tf.Variable(tf.random_normal([dim, 1]))))
q_b_0 = Normal(loc=tf.Variable(tf.random_normal([dim])),
               scale=tf.nn.softplus(tf.Variable(tf.random_normal([dim]))))
q_b_1 = Normal(loc=tf.Variable(tf.random_normal([dim])),
               scale=tf.nn.softplus(tf.Variable(tf.random_normal([dim]))))
q_b_2 = Normal(loc=tf.Variable(tf.random_normal([1])),
               scale=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))

# INFERENCE B
# this will take ten minutes or longer
inference = ed.KLqp(latent_vars={W_0: q_W_0, b_0: q_b_0,
                                 W_1: q_W_1, b_1: q_b_1,
                                 W_2: q_W_2, b_2: q_b_2},
                    data={x: x_train, y: y_train})
inference.run(n_samples=10, n_iter=100000)

# CRITICISM B
plt.scatter(x_train, y_train, s=20.0);  # blue
plt.scatter(x_test, y_test, s=20.0,  # red
            color=sns.color_palette().as_hex()[2]);

xp = tf.placeholder(tf.float32, [1000, D])
[plt.plot(np.linspace(-4.0, 4.0, 1000),
          sess.run(neural_network_with_3_layers(xp,
                                                q_W_0, q_W_1, q_W_2,
                                                q_b_0, q_b_1, q_b_2),
                   {xp: np.linspace(-4.0, 4.0, 1000)[:, np.newaxis]}),
          color='black', alpha=0.1)
 for _ in range(50)];

plt.plot(np.linspace(-4.0, 4.0, 1000),
         sess.run(target_function(xp),  # blue
                  {xp: np.linspace(-4.0, 4.0, 1000)[:, np.newaxis]}));

xp = tf.placeholder(tf.float32, [Np, D])
y_post = Normal(loc=neural_network_with_3_layers(xp,
                                                 q_W_0, q_W_1, q_W_2,
                                                 q_b_0, q_b_1, q_b_2),
                scale=tf.ones(Np) * 0.1)

print("Mean squared error on test data:")
print(ed.evaluate('mean_squared_error', data={xp: x_test, y_post: y_test}))

print("Mean absolute error on test data:")
print(ed.evaluate('mean_absolute_error', data={xp: x_test, y_post: y_test}))



