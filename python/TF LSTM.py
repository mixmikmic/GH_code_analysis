from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np

import basic as b

import tensorflow as tf

name = 'may21_2'
m = b.TSModel(name=name, batchsize=1, timesteps=b.lstm_timesteps, feed_state=False)

ctx = m.graph.as_default()
ctx.__enter__()

sess = tf.InteractiveSession()

saver = tf.train.Saver()
saver.restore(sess, "checkpoints/" + name)

from singen import SinGen

g = SinGen(timesteps=b.lstm_timesteps, batchsize=1)

x, y = g.batch()

y_ = sess.run(m.output, feed_dict={m.input: x, m.labels: y})

m.input

m.output

x.shape

y.shape

b.lstm_timesteps

m.timesteps

y_



