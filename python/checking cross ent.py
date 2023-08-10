import numpy as np
import tensorflow as tf

z = np.array((5,6,10)) 
ps = np.exp(z) / np.sum(np.exp(z))
ps

-np.log(ps[2])

z = tf.constant((5,6,10), dtype='float32')
y = tf.constant(2, dtype='int32')
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(z,y)
with tf.Session() as sess:
    print(sess.run(loss))

