import tensorflow as tf
import numpy as np

a = tf.constant('Hello')
b = tf.constant(' World!')
c = a+b

print(c)

with tf.Session() as sess: 
    runop = sess.run(c)
    
print(runop)

c = tf.add(a,b)

with tf.Session() as sess:
    runop = sess.run(c)
    
print(runop)

mat_a = tf.constant(np.arange(1,12, dtype=np.int32), shape=[2,2,3])
mat_b = tf.constant(np.arange(12,24, dtype=np.int32), shape=[2,3,2])
mul_c = tf.matmul(mat_a, mat_b)

with tf.Session() as sess:
    runop = sess.run(mul_c)
    
print(runop)







