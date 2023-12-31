from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal, Poisson

x_train = np.load('data/celegans_brain.npy')

N = x_train.shape[0]  # number of data points
K = 3  # latent dimensionality

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

# Alternatively, run
# qz = Normal(loc=tf.Variable(tf.random_normal([N * K])),
#             scale=tf.nn.softplus(tf.Variable(tf.random_normal([N * K]))))
# inference = ed.KLqp({z: qz}, data={x: x_train})

inference.run(n_iter=2500)

