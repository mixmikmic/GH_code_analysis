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

import numpy as np

x_data = np.random.randn(2000,3)
w_real = [0.4, 0.6, 0.2]
b_real = -0.3 
noise = np.random.randn(1,2000)*0.1 
y_data = np.matmul(w_real, x_data.T)+b_real+noise

num_iters = 10 
g = tf.Graph()
wb = []

with g.as_default():
    x = tf.placeholder(tf.float32, shape=[None, 3])
    y_true = tf.placeholder(tf.float32, shape=None)
    
    with tf.name_scope('inference') as scope:
        w = tf.Variable([[0,0,0]], dtype=tf.float32, name='W')
        b = tf.Variable(0, dtype=tf.float32, name='b')
        y_pred = tf.matmul(w, tf.transpose(x))+b
        
    with tf.name_scope('loss') as scope:
        loss = tf.reduce_mean(tf.square(y_true-y_pred))
        
    with tf.name_scope('training') as scope:
        lr = 0.5
        optimizer = tf.train.GradientDescentOptimizer(lr)
        train = optimizer.minimize(loss)
        
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        for step in range(num_iters):
            sess.run(train, {x:x_data, y_true:y_data})
            if(step%5==0):
                print(step, sess.run([w,b]))
                wb.append(sess.run([w,b]))
                
        print(10, sess.run([w,b]))

def sigmoid(x):
    return 1/(1+np.exp(-x))

x_data = np.random.randn(20000,3)
w_real = [0.4, 0.6, 0.2]
b_real = -0.3
wb = np.matmul(w_real,x_data.T)+b_real

y_data_bef_noise = sigmoid(wb)
y_data = np.random.binomial(1, y_data_bef_noise)

num_iters = 50
g = tf.Graph()
wb = []

with g.as_default():
    x = tf.placeholder(tf.float32, shape=[None, 3])
    y_true = tf.placeholder(tf.float32, shape=None)
    
    with tf.name_scope('inference') as scope:
        w = tf.Variable([[0,0,0]], dtype=tf.float32, name='W')
        b = tf.Variable(0, dtype=tf.float32, name='b')
        y_pred = tf.matmul(w, tf.transpose(x))+b
        
    with tf.name_scope('loss') as scope:
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        loss = tf.reduce_mean(loss)
        
    with tf.name_scope('training') as scope:
        lr = 0.5
        optimizer = tf.train.GradientDescentOptimizer(lr)
        train = optimizer.minimize(loss)
        
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        
        for step in range(num_iters):
            sess.run(train, {x:x_data, y_true:y_data})
            if(step%5==0):
                print(step, sess.run([w,b]))
                wb.append(sess.run([w,b]))
                
        print(50, sess.run([w,b]))











