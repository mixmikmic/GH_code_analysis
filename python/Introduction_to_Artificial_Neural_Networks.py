import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris = load_iris()
X = iris.data[:,(2,3)]
y = (iris.target==0).astype(np.int)

per_clf = Perceptron(random_state=42)
per_clf.fit(X, y)
y_pred = per_clf.predict([[2,0.5]])

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
X = mnist.data
y = mnist.target.astype(np.int)
X_train = X[:60000]
y_train = y[:60000]
X_test = X[60000:]
y_test = y[60000:]

import tensorflow as tf
feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)

dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[300, 100], n_classes=10, feature_columns=feature_columns)
dnn_clf.fit(x=X_train, y=y_train, batch_size=50, steps=10000)

from sklearn.metrics import accuracy_score
y_pred = list(dnn_clf.predict(X_test))
accuracy_score(y_pred, y_test)

dnn_clf.evaluate(X_test, y_test)

import tensorflow as tf
n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int32, shape=(None), name='y')

def neuron_layer(X, n_neurons, name, activition=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name='weights')
        b = tf.Variable(tf.zeros([n_neurons]), name='biases')
        z = tf.matmul(X, W) + b
        if activition == 'relu':
            return tf.nn.relu(z)
        else:
            return z

with tf.name_scope('dnn'):
    hidden1 = neuron_layer(X, n_hidden1, "hidden1", 'relu')
    hidden2 = neuron_layer(hidden1, n_hidden2, 'hidden2', 'relu')
    logits = neuron_layer(hidden2, n_outputs, 'outputs')

from tensorflow.contrib.layers import fully_connected
with tf.name_scope('dnn'):
    hidden1 = fully_connected(X, n_hidden1, scope='hidden1')
    hidden2 = fully_connected(hidden1, n_hidden2, scope='hidden2')
    logits = fully_connected(hidden2, n_outputs, scope='outputs', activation_fn=None)

with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name='loss')

learning_rate = 0.01
with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuary = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
data = mnist.data
target = mnist.target.astype(np.int)
X_train = data[:60000]
y_train = target[:60000].astype(np.int32)
X_test = data[60000:]
y_test = target[60000:].astype(np.int32)

def fetch_batch(epoch, batch, batch_size):
    np.random.seed(epoch+batch+42)
    indices = np.random.randint(60000, size=batch_size)
    X_batch = X_train[indices]
    y_batch = y_train[indices]
    return X_batch, y_batch

n_epoch = 10
batch_size = 50
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epoch):
        for batch in range(60000//batch_size):
            X_batch, y_batch = fetch_batch(epoch, batch, batch_size)
            sess.run(training_op, feed_dict={X:X_batch, y:y_batch})
        acc_train = accuary.eval(feed_dict={X:X_batch, y:y_batch})
        acc_test = accuary.eval(feed_dict={X:X_test, y:y_test})
        print(epoch, "train accuary:", acc_train, 'Test accuary:', acc_test)
    save_path = saver.save(sess, './my_model_final.ckpt')

with tf.Session() as sess:
    saver.restore(sess, './my_model_final.ckpt')
    Z = logits.eval(feed_dict={X:X_test})
    y_pred = np.argmax(Z, axis=1)

accuracy_score(y_pred, y_test)

