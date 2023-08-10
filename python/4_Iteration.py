import tensorflow as tf

session = tf.InteractiveSession()

x = tf.constant(list(range(10)))

print(x.eval())

session.close()

import resource
print("{} Kb".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

import numpy as np
session = tf.InteractiveSession()

X = tf.constant(np.eye(10000))
Y = tf.constant(np.random.randn(10000, 300))

print("{} Kb".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

Z = tf.matmul(X, Y)

Z.eval()

print("{} Kb".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

session.close()

