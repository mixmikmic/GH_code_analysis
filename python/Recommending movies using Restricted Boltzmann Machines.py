import tensorflow as tf

import pandas as pd

movies = pd.read_csv("../data/intermediate/movies.csv", index_col=0)

movies.head()

import scipy.io

R = scipy.io.mmread("../data/intermediate/user_movie_ratings.mtx").tocsr()


print ('{0}x{1} user by movie matrix'.format(*R.shape))

from __future__ import division

n_visible, n_hidden = len(movies), 20


graph = tf.Graph()

with graph.as_default():
    v_bias = tf.placeholder(tf.float32, [n_visible])
    h_bias = tf.placeholder(tf.float32, [n_hidden])
    W = tf.placeholder(tf.float32, [n_visible, n_hidden])
    
    # visible to hidden pass
    v_1 = tf.placeholder(tf.float32, [None, n_visible])
    h_1_ = tf.sigmoid(tf.matmul(v_1, W) + h_bias)
    h_1 = tf.nn.relu(tf.sign(h_1_ - tf.random_uniform(tf.shape(h_1_))))
    
    
    # hidden to visible pass
    v_2_ = tf.sigmoid(tf.matmul(h_1, tf.transpose(W)) + v_bias)
    v_2 = tf.nn.relu(tf.sign(v_2_ - tf.random_uniform(tf.shape(v_2_))))
    h_2 = tf.nn.sigmoid(tf.matmul(v_2, W) + h_bias)
    
    # Learning rate
    lr = 0.01
    W_gradient_1 = tf.matmul(tf.transpose(v_1), h_1)
    W_gradient_2 = tf.matmul(tf.transpose(v_2), h_2)
    
    contrastive_divergence = ( W_gradient_1 - W_gradient_2 ) / tf.to_float(tf.shape(v_1)[0])
    
    # parameter updates
    W_update = W + lr * contrastive_divergence
    v_bias_update = v_bias + lr * tf.reduce_mean(v_1 - v_2, 0)
    h_bias_update = h_bias + lr * tf.reduce_mean(h_1 - h_2, 0)
    
    # error metrics
    mae = tf.reduce_mean(tf.abs(v_1 - v_2))
    rmse = tf.sqrt(tf.reduce_mean(tf.square(v_1 - v_2)))
    

import numpy as np


n_epoch = 20
batch_size = 100
current_W = np.zeros([n_visible, n_hidden], np.float32)
current_v_bias = np.zeros([n_visible], np.float32)
current_h_bias = np.zeros([n_hidden], np.float32)


# split into train and test
train_R = R[0:4500]
test_R = R[4500:]

errors = []

with tf.Session(graph=graph) as sess:
    tf.initialize_all_variables().run()
    for epoch in range(n_epoch):
        for start in range(0, train_R.shape[0]-batch_size, batch_size):
            end = start + batch_size
            end = start + batch_size
            batch = train_R[start:end].todense()
            feed_dict = { v_1: batch, W: current_W, v_bias: current_v_bias, h_bias: current_h_bias }
            updates = [W_update, v_bias_update, h_bias_update]
            current_W, current_v_bias, current_h_bias = sess.run(updates, feed_dict=feed_dict)
        
        feed_dict = { v_1: test_R.todense(), W: current_W, v_bias: current_v_bias, h_bias: current_h_bias }
        mean_average_error, root_mean_squared_error = sess.run([mae, rmse], feed_dict=feed_dict)
        current_error = { "MAE": mean_average_error, "RMSE": root_mean_squared_error }
        
        print "MAE = {MAE:10.9f}, RMSE = {RMSE:10.9f}".format(**current_error)
        errors.append(current_error)

np.save("../models/W.npy", current_W)
np.save("../models/v_bias.npy", current_v_bias)
np.save("../models/h_bias.npy", current_h_bias)

import numpy as np

current_W = np.load("../models/W.npy")
current_v_bias = np.load("../models/v_bias.npy")
current_h_bias = np.load("../models/h_bias.npy")

import tensorflow as tf
from IPython.display import display, HTML

graph = tf.Graph()

with graph.as_default():
    v_bias = tf.placeholder(tf.float32, [n_visible])
    h_bias = tf.placeholder(tf.float32, [n_hidden])
    W = tf.placeholder(tf.float32, [n_visible, n_hidden])
    v_1 = tf.placeholder(tf.float32, [None, n_visible])
    
    
    h_1 = tf.nn.sigmoid(tf.matmul(v_1, W) + h_bias)
    v_2 = tf.nn.sigmoid(tf.matmul(h_1, tf.transpose(W)) + v_bias)

current_user = R[4500].todense()
recommendations = movies.copy(deep=True)
recommendations["Ratings"] =  current_user[0].T
HTML("<h3> Rated movies </h3>")
display(recommendations.sort_values(by=["Ratings"], ascending = False).head())


print ("current_user = {0}".format(current_user))
with tf.Session(graph=graph) as sess:
    tf.initialize_all_variables().run()
    feed_dict = { v_1: current_user, W: current_W, h_bias: current_h_bias }
    h1 = sess.run(h_1, feed_dict=feed_dict)
    feed_dict = { h_1: h1, W: current_W, v_bias: current_v_bias }
    v2 = sess.run(v_2, feed_dict=feed_dict)
    recommendations["Score"] = v2[0] * 5.0
    HTML("<h3> Recommended movies </h3>")
    display(recommendations.sort_values(by=["Score"], ascending = False).head())

