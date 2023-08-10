import numpy as np

import tensorflow as tf
from tensorflow.contrib import keras
from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import regularizers
from tensorflow.contrib.keras import optimizers

tf.set_random_seed(42)

n_in = 3
n_factors = 2

in_ = layers.Input(shape=(1,), dtype='int64')

emb = layers.Embedding(n_in, n_factors, input_length=1)(in_)

model = models.Model(in_, emb)

model.predict(np.array([0, 1, 2]))

model.predict(np.array([0, 1, 2]))

n_in = 3
n_factors = 2

in_ = layers.Input(shape=(1,), dtype='int64')

weights = np.array([ [0, 0], [-1, -1], [-2, -2] ])

weights

pretrained_emb = layers.Embedding(n_in, n_factors, input_length=1, weights=[weights])(in_)

model = models.Model(in_, pretrained_emb)

model.predict(np.array([0]))

model.predict(np.array([1, 0, 2]))

n_factors = 2

n_in1 = 3
n_in2 = 5

in1 = layers.Input(shape=(1,), dtype='int64')
in2 = layers.Input(shape=(1,), dtype='int64')

emb1 = layers.Embedding(n_in1, n_factors, input_length=1)(in1)
emb2 = layers.Embedding(n_in2, n_factors, input_length=1)(in2)

model = models.Model([in1, in2], [emb1, emb2])

model.predict([np.array([2]), np.array([4])])



