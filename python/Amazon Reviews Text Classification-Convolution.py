from __future__ import print_function 
import numpy as np
import tensorflow as tf 

import os
import sys

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Lambda
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import SimpleRNN, GRU, LSTM, Bidirectional

import keras.backend as K

f = np.load('data_and_embedding.npz')

num_words = int(f['num_words'])
embedding_dim = int(f['embedding_dim'])
max_sequence_length = int(f['max_sequence_length'])

data = f['data']
labels = f['labels']

embedding_matrix = f['embedding_matrix']

validation_split = 0.2 
epochs = 60

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(validation_split * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

embedding_layer = Embedding(num_words,
                            embedding_dim,
                            weights=[embedding_matrix],
                            input_length=max_sequence_length,
                            trainable=False)

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
# preds = embedded_sequences
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)
print(x.shape)
x = Flatten()(x)
print(x.shape)
x = Dense(128, activation='relu')(x)
print(x.shape)
preds = Dense(6, activation='softmax')(x)

model_conv = Model(sequence_input, preds)

model_conv.summary()

model_conv.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

import time 
start_time = time.time()

model_conv.fit(x_train, y_train,
          batch_size=128,
          epochs=epochs,
          validation_data=(x_val, y_val))

print("Training time: ", time.time() - start_time)

model_conv.save('models/Conv.h5')

