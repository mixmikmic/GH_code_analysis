import tensorflow as tf
import numpy as np

X = [[0, 0],[0, 1],[1, 0],[1, 1]]
Y = [[0], [1], [1], [0]]

N_STEPS = 14000
N_EPOCH = 4000
N_TRAINING = len(X)
LEARNING_RATE = 0.05
N_INPUT_NODES = 2
N_HIDDEN_NODES = 4
N_OUTPUT_NODES  = 1

x_ = tf.placeholder(tf.float32, shape=[N_TRAINING, N_INPUT_NODES], name="x-input")
y_ = tf.placeholder(tf.float32, shape=[N_TRAINING, N_OUTPUT_NODES], name="y-input")

w1 = tf.Variable(tf.random_uniform([N_INPUT_NODES,N_HIDDEN_NODES], -1, 1), name="theta1")
w2 = tf.Variable(tf.random_uniform([N_HIDDEN_NODES,N_OUTPUT_NODES], -1, 1), name="theta2")

bias1 = tf.Variable(tf.zeros([N_HIDDEN_NODES]), name="bias1")
bias2 = tf.Variable(tf.zeros([N_OUTPUT_NODES]), name="bias2")

layer1_output = tf.sigmoid(tf.matmul(x_, w1) + bias1)

output = tf.sigmoid(tf.matmul(layer1_output, w2) + bias2)

cost = - tf.reduce_mean( (y_ * tf.log(output)) + (1 - y_) * tf.log(1.0 - output) )

train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

init = tf.global_variables_initializer()
#saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    for i in range(N_STEPS):
        sess.run(train_step, feed_dict={x_: X, y_: Y})
        if i % N_EPOCH == 0:
            #print('Batch ', i)
            print('Output', sess.run(output, feed_dict={x_: X, y_: Y}))
            print('Cost', sess.run(cost, feed_dict={x_: X, y_: Y}))
            print ('\n')
    #save_path = saver.save(sess, "/tmp/model.ckpt")
    #print("Model saved in file: %s" % save_path)

