import sys
print(sys.version)

print(sys.path)

import pyodbc
from pandas import *
import pandas as pd
import numpy as np
import json
from pandas.io.json import json_normalize
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# initialize the connection
conn = pyodbc.connect("DSN=drill64", autocommit=True)
# Set unicode options so the ODBC driver returns column names and table contents as ASCII strings
conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf-32le', to=str)
conn.setdecoding(pyodbc.SQL_WMETADATA, encoding='utf-32le', to=str)
cursor = conn.cursor()

s = "select * from dfs.`/apps/mqtt_records`"
df = pandas.read_sql(s, conn)
pd.set_option('display.max_columns', 200)
# infer data types for each column
df=df.apply(pd.to_numeric, errors='ignore')
print ("Loaded " + str(len(df.index)) + " rows and " + str(len(df.columns)) + " columns.")
df.head(5)

df = df.sort_values(by=['timestamp'])
df['timestamp']=pd.to_datetime(df['timestamp'], unit='s')
ts = pd.Series(df['BuildingPower'].values, index=df['timestamp'])
ts.head(100).plot(c='b', title="Time series data")
plt.show()
ts.head(10)

TS = np.array(ts[0:801])
num_periods = 20
f_horizon = 1

x_data = TS[:(len(TS)-(len(TS) % num_periods))]
x_batches = x_data.reshape(-1, 20, 1)

y_data = TS[1:(len(TS)-(len(TS) % num_periods))+f_horizon]
y_batches = y_data.reshape(-1, 20, 1)
print (len(x_batches))
print (x_batches.shape)
print (y_batches.shape)

def test_data(series,forecast,num_periods):
    test_x_setup = TS[-(num_periods + forecast):]
    testX = test_x_setup[:num_periods].reshape(-1, 20, 1)
    testY = TS[-(num_periods):].reshape(-1, 20, 1)
    return testX,testY
X_test, Y_test = test_data(TS,f_horizon,num_periods)
print (X_test.shape)

import tensorflow as tf
import os
import shutil
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib.learn.python.learn import learn_runner
import tensorflow.contrib.metrics as metrics
import tensorflow.contrib.rnn as rnn

tf.reset_default_graph()
# num_periods = 20
inputs = 1
hidden = 100
output = 1

x = tf.placeholder(tf.float32, [None, num_periods, inputs])
y = tf.placeholder(tf.float32, [None, num_periods, output])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden, activation=tf.nn.relu)
rnn_output, states = tf.nn.dynamic_rnn(basic_cell, x, dtype=tf.float32)
learning_rate = 0.001
stacked_rnn_output = tf.reshape(rnn_output, [-1, hidden])
stacked_outputs = tf.layers.dense(stacked_rnn_output, output)
outputs = tf.reshape(stacked_outputs, [-1, num_periods, output])

loss = tf.reduce_sum(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
training_op = optimizer.minimize(loss)
init = tf.global_variables_initializer()

epochs = 1000

with tf.Session() as sess:
    init.run()
    for ep in range(epochs):
        sess.run(training_op, feed_dict={x: x_batches, y: y_batches})
        if ep % 100 == 0:
            mse = loss.eval(feed_dict={x: x_batches, y: y_batches})
            print (ep, "\tMSE:", mse)
            
    y_pred = sess.run(outputs, feed_dict={x: X_test})

plt.title("Forecast vs Actual", fontsize = 14)
timestamps = df['timestamp'][-len(pd.Series(np.ravel(Y_test))):].values
plt.plot(timestamps, pd.Series(np.ravel(Y_test)), "b-", markersize = 10)
plt.plot(timestamps, pd.Series(np.ravel(Y_test)), "bo", markersize = 10, label="Actual")
plt.plot(timestamps, pd.Series(np.ravel(y_pred)), "r-", markersize = 10)
plt.plot(timestamps, pd.Series(np.ravel(y_pred)), "r.", markersize = 10, label="Forecast")
plt.legend(loc="upper left")
plt.xlabel("Time")

plt.show()



