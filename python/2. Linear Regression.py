import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.01
training_epochs = 1000
display_step = 50

train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]

#Declare input data type
x=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)

#Declare trainable parameters
randomizer=np.random
W=tf.Variable(randomizer.randn(), tf.float32, name="weight")
b=tf.Variable(randomizer.randn(), tf.float32, name="bias")

#Declare model
linear_model=tf.add(tf.multiply(W,x),b) #depending on your Tensorflow version, you might need to change tf.mul to tf.multiply

#Declare cost function
loss=tf.reduce_sum(tf.square(linear_model-y)) #sum of squares
loss_=tf.reduce_sum(tf.pow(linear_model-y,2))/(2*n_samples) #mean squared error

optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_)

#Initialize variables
init = tf.global_variables_initializer()

#Perform learning
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        for (x_, y_) in zip(train_X, train_Y):
            sess.run(optimizer, {x:x_, y:y_})

    #Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(loss_, feed_dict={x: train_X, y:train_Y})
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c),                 "W=", sess.run(W), "b=", sess.run(b)

    print "Optimization Finished!"
    training_cost = sess.run(loss_, feed_dict={x: train_X, y: train_Y})
    print "Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n'

     #Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()

