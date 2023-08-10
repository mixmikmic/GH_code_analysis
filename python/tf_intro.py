from __future__ import print_function

import numpy as np
import tensorflow as tf



tf.initialize_all_variables()

with tf.Session():
    inp1 = tf.constant([1, 1, 1, 1])
    inp2 = tf.constant([2, 2, 2, 2])
    output = tf.add(inp1, inp2)
    result = output.eval()
    print(result)

with tf.Session():
    input1 = tf.constant(1.0, shape=[2, 3])
    input2 = tf.constant(np.reshape(np.arange(1.0, 7.0, dtype=np.float32), (2, 3)))
    output = tf.add(input1, input2)
    print(output.eval())

with tf.Session():
    input_features = tf.constant(np.reshape([1, 0, 0, 1], (1, 4)).astype(np.float32))
    weights = tf.constant(np.random.randn(4, 2).astype(np.float32))
    output = tf.matmul(input_features, weights)
    print("Input:")
    print(input_features.eval())
    print("Weights:")
    print(weights.eval())
    print("Output:")
    print(output.eval())

# An integer parameter
N = tf.placeholder('int64', name="n")

# Sum of squares of N integers
result = tf.reduce_sum(tf.range(N)**2)

with tf.Session() as sess:
    print(result.eval({N: 10**8}))



