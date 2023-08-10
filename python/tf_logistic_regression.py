import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

get_ipython().run_line_magic('matplotlib', 'inline')

# load only two-digits dataset by passing the argument '2'
digits = load_digits(2)
X = digits.data
Y = digits.target

print("Y :")
print("The shape of Y is : ", Y.shape)
print("1st 10 elements of Y is : ", Y[:10])

print("X: ")
print("The shape of X is : ", X.shape)
print("Each X is of shape : ", X[0].shape)
print("An example of X is : \n", X[2])
print("Its respective label is : ", Y[2])

plt.imshow(X[3].reshape([8,8]), cmap="gray")

plt.imshow(X[0].reshape([8,8]), cmap="gray")

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, random_state=42)

# create global variables for weights and bias which have to be updated iteratively
weights = tf.Variable(dtype=tf.float32, initial_value=2*np.random.random((X.shape[1], 1))*0.001, name='weights')
b = tf.Variable(dtype=tf.float32, initial_value=1, name='b')

print(weights)
print(b)

# create a dummy value representing x and y data
input_X = tf.placeholder(dtype=tf.float32, name='input_x')
input_Y = tf.placeholder(dtype=tf.float32, name='input_y')

print(input_X)
print(input_Y)

predicted_y = tf.squeeze(tf.nn.sigmoid(tf.add( tf.matmul(input_X, weights) , b)))
print(predicted_y)

loss = -tf.reduce_mean(input_Y*tf.log(predicted_y) + (1-input_Y)*tf.log(1-predicted_y))
print(loss)

optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
print(optimizer)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(5):
        sess.run(optimizer, {input_X: X_train, input_Y: Y_train})
        loss_i = sess.run(loss, {input_X: X_train, input_Y: Y_train})
        print("loss at iter %i: %.4f" % (i, loss_i))
        print("train auc:", roc_auc_score(Y_train, sess.run(predicted_y, {input_X:X_train})))
        print("test auc:", roc_auc_score(Y_test, sess.run(predicted_y, {input_X:X_test})))

