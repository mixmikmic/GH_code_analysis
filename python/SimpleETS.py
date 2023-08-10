import numpy as np
import tensorflow as tf

# Meta-parameters and debugging knobs
learning_rate = 0.01
training_epochs = 1000
display_step = 50

# Test data
y = np.asarray([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0])
num_steps = y.shape[0]

# Input data placeholders
data_in = tf.placeholder('float')
data_out = tf.placeholder('float')

# ETS params
initial = tf.Variable(0.0, name = 'initial', dtype = tf.float32)
alpha = tf.Variable(0.5, name = 'alpha', dtype = tf.float32)

# Definition of the ETS update
def update(y, level):
    return level, level + alpha * (y - level)

# Unrolled ETS loop
outputs = []
level = initial
for time_step in range(num_steps):
    output, level = update(data_in[time_step], level)
    outputs.append(output)

# Mean squared error
cost = tf.reduce_sum(tf.pow(tf.pack(outputs) - data_out, 2))

# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Fit the data.    
    for epoch in range(training_epochs):
        sess.run(optimizer, feed_dict={data_in: y, data_out: y})

        # Display logs per epoch step
        if (epoch + 1) % display_step == 0:
            c = sess.run(cost, feed_dict={data_in: y, data_out: y})
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c),                 "initial=", sess.run(initial), "alpha=", sess.run(alpha)

    print "Optimization Finished!"
    training_cost = sess.run(cost, feed_dict={data_in: y, data_out: y})
    print "Training cost=", training_cost, "initial=", sess.run(initial), "alpha=", sess.run(alpha), '\n'



