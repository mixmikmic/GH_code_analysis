import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import os, sys
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf

import tf_threads
tfconfig = tf_threads.limit(tf, 2)

from sklearn.preprocessing import scale, LabelBinarizer
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import datasets

iris = datasets.load_iris()

X = iris["data"]

y = iris["target"]

X.shape

y.shape

X_train, y_train, X_test, y_test = train_test_split(X, y, test_size = 0.05)

print("Class distribution:", {i: np.sum(y == i) for i in np.unique(y)})

class MultilayerPerceptron(object):
    def __init__(self, session, features, labels):
        hidden_layer = tf.layers.dense(features, 4, tf.tanh)
        hidden_layer2 = tf.layers.dense(hidden_layer, 3, tf.tanh)
        predictions = tf.layers.dense(hidden_layer2, 3, tf.sigmoid)

        # Loss function
        loss = tf.losses.mean_squared_error(labels, tf.squeeze(predictions))

        # An optimizer defines the operation for updating parameters within the model.
        optimizer = tf.train.AdamOptimizer(learning_rate=0.03)

        # Training is defined as minimizing the loss function using gradient descent.
        training = optimizer.minimize(loss)
        
        self.context = [session, training, loss, predictions]
        
    def fit(self, X_train, y_train, N_BATCH=32):
        sess, training, loss, _  = self.context
        label_encoding=LabelBinarizer()
        label_encoding.fit(y)
        
        training_loss = []
        for epoch in range(25):
            epoch_loss = []
            for i in range(0, X_train.shape[0], N_BATCH):
                _, batch_loss = sess.run([training, loss], feed_dict={
                    features: X_train[i: i + N_BATCH],
                    labels: label_encoding.transform(y_train[i: i + N_BATCH])
                })
                epoch_loss.append(batch_loss)
            training_loss.append(np.mean(epoch_loss))
            print('epoch', epoch, 'loss:', training_loss[-1])
        self.training_loss = training_loss
        self.label_encoding = label_encoding
        
    def predict(self, X_test, N_BATCH=32):
        sess, _, _, predictions  = self.context
        
        y_pred = []
        for i in range(0, X_test.shape[0], N_BATCH):
            batch_prediction = sess.run(predictions, feed_dict={
                features: X_test[i: i + N_BATCH]
            })
            class_probablity = self.label_encoding.inverse_transform(np.exp(batch_prediction))
            y_pred.extend(class_probablity)
        return np.array(y_pred)

with tf.Session(config=tfconfig) as sess:
    features = tf.placeholder("float", (None, 4))
    labels = tf.placeholder("float", (None, 3))
    mlp = MultilayerPerceptron(sess, features, labels)
    sess.run(tf.global_variables_initializer())
    mlp.fit(X_train, y_train)
    
    plt.figure(figsize=(6,4))
    plt.title('loss')
    plt.plot(range(len(mlp.training_loss)), mlp.training_loss)
    
    plt.figure(figsize=(4,4))
    y_pred = mlp.predict(X_test)
    print('accuracy', accuracy_score(y_test, y_pred))
    plt.imshow(confusion_matrix(y_test, y_pred))



