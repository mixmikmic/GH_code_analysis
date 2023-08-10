import tensorflow as tf
tf.set_random_seed(777)

# set data
x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

# prepare placeholders
n_features = 4
n_classes = 3
X = tf.placeholder(tf.float32, [None, n_features])
Y = tf.placeholder(tf.float32, [None, n_classes])

W = tf.Variable(tf.random_normal([n_features, n_classes], name='weight'))
b = tf.Variable(tf.random_normal([n_classes], name='bias'))

# out hypothesis
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# the cross entrophy loss
cost = tf.reduce_mean( -tf.reduce_sum(Y * tf.log(hypothesis), axis=1) )

# our optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# start training
training_size = 2000
print_step = 200
with tf.Session() as sess:
    # initialize variables
    sess.run(tf.global_variables_initializer())
    
    for step in range(training_size):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        
        if step % print_step == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}))
    
    # Testing & One-hot encoding
    print('--------------')
    a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9]]})
    print(a, sess.run(tf.arg_max(a, 1)))
    
    print('--------------')
    b = sess.run(hypothesis, feed_dict={X: [[1, 3, 4, 3]]})
    print(b, sess.run(tf.arg_max(b, 1)))
    
    print('--------------')
    c = sess.run(hypothesis, feed_dict={X: [[1, 1, 0, 1]]})
    print(c, sess.run(tf.arg_max(c, 1)))
    
    print('--------------')
    all = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]]})
    print(all, sess.run(tf.arg_max(all, 1)))
    

