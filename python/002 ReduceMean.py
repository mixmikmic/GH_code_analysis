import tensorflow as tf
import numpy as np

init_op = tf.global_variables_initializer()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.001, allow_growth=True)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(init_op)

data = np.array([[[1, 2, 3, 4]], 
                 [[5, 6, 7, 8]], 
                 [[0, 10, 20, 30]]])

inputs = tf.placeholder('float32', shape=(None, None, 4))
mean1 = tf.reduce_mean(inputs)
mean2 = tf.reduce_mean(inputs, reduction_indices=[0])
mean3 = tf.reduce_mean(inputs, reduction_indices=[1])
mean4 = tf.reduce_mean(inputs, reduction_indices=[2])
mean5 = tf.reduce_mean(inputs, reduction_indices=[1, 2])

init_op = tf.global_variables_initializer()
sess.run(init_op)

sess.run([mean1, mean2, mean3, mean4, mean5], feed_dict={inputs: data})

data.mean()

sess.run(mean1, feed_dict={inputs: data})

data.mean(axis=0)

sess.run(mean2, feed_dict={inputs: data})

data.mean(axis=2)

sess.run(mean4, feed_dict={inputs: data})

data.mean(axis=2).mean(axis=1)

sess.run(mean5, feed_dict={inputs: data})

