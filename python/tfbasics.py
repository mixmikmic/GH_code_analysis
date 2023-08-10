get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Define C=B*A in a symbolic way
A = tf.Variable(tf.ones([10]))
B = tf.constant(np.ones(10)*2, tf.float32)
C = tf.multiply(A, B)
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    # initialize variables
    sess.run(init)
    # run the graph and evaluate C
    c = sess.run([C])
    print('c:', c)

# Generate ground truth 100 x, y data points in NumPy, y = 3.0 * x + 1.0
# Regress for W and b that compute y_data = W * x_data + b
x_data = np.random.rand(100).astype(np.float32)
y_data = 3.0 * x_data + 1.0
plt.plot(x_data, y_data)

# define trainable variables
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))

# define graph operations
y = tf.multiply(W, x_data) + b

# define loss, L2
loss = tf.reduce_mean(tf.square(y - y_data))

# define optimizer for training
train_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)

# define the operation that initializes variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    # initialization
    sess.run(init)

    # starting training
    training_iters = 100
    for step in range(training_iters):
        if step % 20 == 0 or (step+1)==training_iters:
            print(step, sess.run(W), sess.run(b))
            
        # run optimizer during training
        _ = sess.run([train_optimizer])

