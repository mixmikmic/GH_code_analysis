import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

# get data
our_data = np.array([[828.659973, 833.450012, 908100, 828.349976, 831.659973],
                     [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
                     [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
                     [816, 820.958984, 1008100, 815.48999, 819.23999],
                     [819.359985, 823, 1188100, 818.469971, 818.97998],
                     [819, 823, 1198100, 816, 820.450012],
                     [811.700012, 815.25, 1098100, 809.780029, 813.669983],
                     [809.51001, 816.659973, 1398100, 804.539978, 809.559998]])

# split data
x_data = our_data[:, 0:-1]
y_data = our_data[:, -1:]

# placeholders
n_features = 4
n_labels = 1
X = tf.placeholder(tf.float32, shape=[None, n_features])
Y = tf.placeholder(tf.float32, shape=[None, n_labels])

# variables
W = tf.Variable(tf.random_normal([n_features, n_labels]), name='weight')
b = tf.Variable(tf.random_normal([n_labels]), name='bias')

# hypothesis
hypothesis = tf.matmul(X, W) + b

# loss
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost)

# train
train_size = 200
print_step = 20
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(train_size):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        
        if step % print_step == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}))
    

