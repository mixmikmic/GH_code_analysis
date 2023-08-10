import tensorflow as tf
import pandas
dataset=pandas.read_csv('dataset.csv')
setx=dataset.values[:,0]
sety=dataset.values[:,1]

# Graph inputs
X = tf.placeholder("float")
Y = tf.placeholder("float")

# model weights
m = tf.Variable(tf.random_normal([1]), name="weight")
c = tf.Variable(tf.random_normal([1]), name="bias")

# a linear model
prediction = tf.add(tf.multiply(X, m), c)

# Mean squared error
cost = tf.reduce_sum(tf.pow(prediction-Y, 2))

# use an optimizer to update the variables.
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    sess.run(init_op)
    # Fit all training data
    for epoch in range(100):
        for (x, y) in zip(setx, sety):
            sess.run(optimizer, feed_dict={X: x, Y: y})
        # Display summaries every 10 epochs
        if epoch % 10 ==0:
            err = sess.run(cost, feed_dict={X: setx, Y:sety})
            print "Epoch:", epoch, "cost=", "{:.9f}".format(err), "m=", sess.run(m), "c=", sess.run(c)
    trained_m=sess.run(m)
    trained_c=sess.run(c)
    print "Optimization Finished!"

import matplotlib.pyplot as plt
plt.plot(setx, trained_m * setx + trained_c, label='Fitted line')
plt.plot(setx, sety, 'ro', label='Data')
plt.legend()
plt.show()

