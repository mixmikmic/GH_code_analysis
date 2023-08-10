from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib

get_ipython().magic('matplotlib inline')

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../data", one_hot = True)

index = 10
tmp = mnist.train.images[index]
tmp = tmp.reshape((28,28))

plt.imshow(tmp, cmap = cm.Greys)
plt.show()
print("One-hot Label for this images = ", end=" ")
onehot_label = mnist.train.labels[index]
print(onehot_label)
print("Index = %d" % np.argmax(onehot_label))

X = tf.placeholder(tf.float32, [None,784])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#model
net = tf.matmul(X, W) + b #logits
Y = tf.nn.softmax(net)

# Define loss and optimizer
Y_ = tf.placeholder(tf.float32, [None, 10])
#loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=net))

#Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.05)
train_step = optimizer.minimize(cross_entropy)
#or train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# % of correct answers found in batch
is_correct = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(10000):
    #load batch of images and correct answers
    batch_X, batch_Y = mnist.train.next_batch(100)
    train_data={X: batch_X, Y_: batch_Y}
    #train
    sess.run(train_step, feed_dict=train_data)
    if i % 100 == 0:
        #success ?
        a,c = sess.run([accuracy,cross_entropy],feed_dict=train_data)
        print("Step : %d acc = %.4f loss = %.4f" % (i,a,c))
    #--- edit
#success on test data?
test_data = {X: mnist.test.images, Y_: mnist.test.labels}
a,c = sess.run([accuracy, cross_entropy], feed_dict=test_data)
print("Test data acc = %.4f loss = %.4f" % (a,c))

print("Test image size")
print(mnist.test.images.shape)

im_test = mnist.test.images[0].reshape([28,28])
plt.imshow(im_test, cmap= cm.Greys)
plt.show()
#feed test again. By interactive sess we can use eval without sess !! easy!?
res = net.eval(feed_dict = {X:[mnist.test.images[0]]})
print("Result size : ")
print(res.shape)
print("Picking up first response ")
print(res[0])
print("Softmax percentage : ")
print(tf.nn.softmax(res[0]).eval())
print("Result are : %d" % np.argmax(res[0]))

wts = W.eval()
wts.shape
for i in range(0,10):
    im = wts.flatten()[i::10].reshape((28,-1))
    plt.imshow(im, cmap = cm.Greys)
    plt.show()



