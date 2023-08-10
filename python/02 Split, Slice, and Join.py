import tensorflow as tf
import numpy as np

from pprint import pprint as pp

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.01, allow_growth=True)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

data = np.array([[[1, 1, 1], [2, 2, 2]],
                 [[3, 3, 3], [4, 4, 4]],
                 [[5, 5, 5], [6, 6, 6]]])


s1 = tf.slice(data, [1, 0, 0], [1, 1, 3])
s2 = tf.slice(data, [1, 0, 0], [1, 2, 3])
s3 = tf.slice(data, [1, 0, 0], [2, 1, 3])

print('S1')
print(s1.eval())
print(data[1:1+1, 0:0+1, 0:0+3])

print('\nS2')
print(s2.eval())
print(data[1:1+1, 0:0+2, 0:0+3])

print('\nS3')
print(s3.eval())
print('-------------')
print(data[1:1+2, 0:0+1, 0:0+3])

data = np.arange(5*30).reshape((5, 30))
print('Data')
print(data)

splitted1 = tf.split(data, [4, 15, 11], axis=1)

print('\nSplitted 01')
print(tf.shape(splitted1[0]).eval()) # 4
print(splitted1[0].eval())

print('\nSplitted 02')
print(tf.shape(splitted1[1]).eval()) # 15
print(splitted1[1].eval())

print('\nSplitted 03')
print(tf.shape(splitted1[2]).eval()) # 11
print(splitted1[2].eval())

splitted = tf.split(data, [1, 2, 2], axis=0)

print('\n\nSplitted 01')
print('Shape:', tf.shape(splitted[0]).eval())
print(splitted[0].eval())
print('numpy')
print(data[0:1])

print('\n\nSplitted 02')
print('Shape:', tf.shape(splitted[1]).eval())
print(splitted[1].eval())
print('numpy')
print(data[1:3])

print('\n\nSplitted 03')
print('Shape:', tf.shape(splitted[2]).eval())
print(splitted[2].eval())
print('numpy')
print(data[3:5])



