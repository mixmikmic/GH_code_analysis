from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from edward.models import Normal, Poisson
from observations import celegans

x_train = celegans("~/data")
#x_train = x_data[:148,:148]
#x_test = x_data[148:296,148:296]

N = x_train.shape[0]  # number of data points
K = 6  # latent dimensionality

z = Normal(loc=tf.zeros([N, K]), scale=tf.ones([N, K]))

# Calculate N x N distance matrix.
# 1. Create a vector, [||z_1||^2, ||z_2||^2, ..., ||z_N||^2], and tile
# it to create N identical rows.
xp = tf.tile(tf.reduce_sum(tf.pow(z, 2), 1, keep_dims=True), [1, N])
# 2. Create a N x N matrix where entry (i, j) is ||z_i||^2 + ||z_j||^2
# - 2 z_i^T z_j.
xp = xp + tf.transpose(xp) - 2 * tf.matmul(z, z, transpose_b=True)
# 3. Invert the pairwise distances and make rate along diagonals to
# be close to zero.
xp = 1.0 / tf.sqrt(xp + tf.diag(tf.zeros(N) + 1e3))

x = Poisson(rate=xp)

inference = ed.MAP([z], data={x: x_train})

#Alternatively, run
qz = Normal(loc=tf.get_variable("qz/loc", [N, K]),
             scale=tf.nn.softplus(tf.get_variable("qz/scale", [N, K])))

inference = ed.KLqp({z: qz}, data={x: x_train})

inference.run(n_iter=2500)

# build posterior predictive after inference: it is
# parameterized by a posterior sample
x_post = ed.copy(x, {z: qz})

# log-likelihood performance
x_test = tf.cast(x_test, tf.float32)
ed.evaluate(['log_loss','log_likelihood'], data={x_post: x_test})

x_post

train_loss = [11977.5,11858.1,12096.3,12232.8,12992.3,13738.7]
train_k = [3,6,9,12,30,60]
test_log_loss = [0.81370974,0.80406153,0.7961016,0.7887505,0.75811666,0.73667634]

plt.plot(train_k,train_loss)
plt.ylabel('train loss')
plt.xlabel('Latent dimension (K)')
plt.show()

plt.plot(train_k,test_log_loss)
plt.ylabel('test log-loss')
plt.xlabel('Latent dimension (K)')
plt.show()

test_loss = []
for i in range(1,100):
    tf.reset_default_graph()
        
    N = x_train.shape[0]  # number of data points
    K = i  # latent dimensionality
    z = Normal(loc=tf.zeros([N, K]), scale=tf.ones([N, K]))
    xp = tf.tile(tf.reduce_sum(tf.pow(z, 2), 1, keepdims=True), [1, N])
    xp = xp + tf.transpose(xp) - 2 * tf.matmul(z, z, transpose_b=True)
    xp = 1.0 / tf.sqrt(xp + tf.diag(tf.zeros(N) + 1e3))
    x = Poisson(rate=xp)
    
    inference = ed.MAP([z], data={x: x_train})
    inference.run(n_iter=100)
    
    info_dict = inference.update()
    test_loss.append(info_dict['loss'])
    ed.get_session().close()

plt.plot(test_loss)

plt.show()



