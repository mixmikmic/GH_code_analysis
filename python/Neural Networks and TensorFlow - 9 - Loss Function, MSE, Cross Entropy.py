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

#with tf.Session() as sess:
#    sess.run(ops)
#    sess.close()

x = tf.placeholder(tf.float32, name='X', shape=(4,9))
w = tf.placeholder(tf.float32, name='W', shape=(9,1))
b = tf.fill((4,1), -1., name='bias')
y = tf.matmul(x,w)+b
s = tf.reduce_max(y)

x_data = np.random.randn(4,9)
w_data = np.random.randn(9,1)

with tf.Session() as sess:
    out_p = sess.run(s, feed_dict={x:x_data, w:w_data})
    
print('Output: {}'.format(out_p))

# X, Y, W, b

x = tf.placeholder(tf.float32, shape=[None, 3])
y_true = tf.placeholder(tf.float32, shape=None)
w = tf.Variable([[0,0,0]], dtype=tf.float32, name='W')
b = tf.Variable(0, dtype=tf.float32, name='b')

# Output

y_pred = tf.matmul(w, tf.transpose(x))+b

# MSE (Mean Squared Error)

loss = tf.reduce_mean(tf.square(y_true-y_pred))

# Cross Entropy

loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
loss = tf.reduce_mean(loss)







