import numpy as np
import pandas as pd
import pickle

from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import activations
from keras import metrics
from keras import regularizers
from keras.callbacks import *

import tensorflow as tf

from dataset import atti_dirigenti

import math

import random

# to make the experimens replicable
random.seed(123456)

from tensorflow.python.client import device_lib

device_lib.list_local_devices()

(x_train, y_train), (x_val, y_val), (x_test, y_test) = atti_dirigenti.load_data(num_words=10000, remove_stopwords=True)

label_index = atti_dirigenti.get_labels()
len(label_index)

from keras.preprocessing import sequence

print('max length of objects {}'.format(max(map(len, x_train))))

maxlen = 100

x_train_pad = sequence.pad_sequences(x_train, maxlen=maxlen)
x_val_pad = sequence.pad_sequences(x_val, maxlen=maxlen)
x_test_pad = sequence.pad_sequences(x_test, maxlen=maxlen)

x_train_pad.shape

def vectorize_sequences(sequences, dimension=10000):
    data = np.zeros((len(sequences), dimension), dtype=np.float16)
    for i, sequence in enumerate(sequences):
        data[i,sequence] = 1.
    return data

dimension = 11000

x_train_vect = vectorize_sequences(x_train, dimension)
x_val_vect = vectorize_sequences(x_val, dimension)
x_test_vect = vectorize_sequences(x_test, dimension)

def to_one_hot(labels):
    results = np.zeros((len(labels), len(set(labels))), dtype=np.float32)
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

y_train_vect = to_one_hot(y_train)
y_val_vect = to_one_hot(y_val)
y_test_vect = to_one_hot(y_test)

y_train_vect.shape

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (6, 4)
plt.rcParams['figure.dpi'] = 150

get_ipython().run_line_magic('matplotlib', 'inline')

def chart_loss(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, 'b+', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yticks(np.arange(0,2, step=0.2))
    plt.xticks(np.arange(0,len(loss), step=1))
    plt.legend()
    plt.show()

def chart_acc(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'b+', label='Training Acc')
    plt.plot(epochs, val_acc, 'b', label='Validation Acc')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Accuray')
    plt.yticks(np.arange(0.4,1, step=0.05))
    plt.xticks(np.arange(0,len(acc), step=1))
    plt.legend()
    plt.show()

def min_loss(history):
    val_loss = history.history['val_loss'] 
    return np.argmin(val_loss) + 1   

def accuracy(history, epoch):
    val_acc = history.history['val_acc']
    return val_acc[epoch-1]

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, mode='min',cooldown=0, min_lr=0)
]

def build_model_dropout(neurons):
    with tf.device('/gpu:0'):
        model = models.Sequential()
        model.add(layers.Dense(neurons, activation='relu', input_shape=(x_train_vect.shape[-1], )))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(neurons, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(neurons, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(len(label_index), activation='softmax'))

        model.compile(optimizer=optimizers.Adam(), 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model_dropout = build_model_dropout(256)
model_dropout.summary()

history_dropout = model_dropout.fit(x=x_train_vect, y=y_train_vect, validation_data=(x_val_vect, y_val_vect),
                   epochs=10, batch_size=256, callbacks=callbacks)

def build_model_cnn(embed_size):
    with tf.device('/gpu:0'):
        model = models.Sequential()
        model.add(layers.Embedding(input_dim=dimension, output_dim=embed_size, input_length=maxlen, name='embed'))
        model.add(layers.Conv1D(16, 5, activation='relu'))
        model.add(layers.MaxPooling1D(5))
        model.add(layers.Conv1D(32, 5, activation='relu'))
        model.add(layers.GRU(64, activation='relu',
                             dropout= 0.5,
                             recurrent_dropout = 0.5,
                             return_sequences=False))
        model.add(layers.Dense(len(label_index), activation='softmax'))

        model.compile(optimizer=optimizers.Adam(), 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

model_cnn = build_model_cnn(128)
model_cnn.summary()

history_cnn = model_cnn.fit(x=x_train_pad, y=y_train_vect,  validation_data=(x_val_pad, y_val_vect),
                   epochs=10, batch_size=256)

def build_model_rnn(embed_size, neurons, bidirectional=False, num_layers=1, 
                    dropout=0.5, recurrent_dropout=0.5, cell_type=layers.GRU):
    with tf.device('/gpu:0'):
        model = models.Sequential()
        model.add(layers.Embedding(input_dim=dimension, output_dim=embed_size, input_length=maxlen, name='embed'))

        def create_layer(return_sequences=True):
            layer = cell_type(neurons, activation='relu',
                                     dropout= dropout,
                                     recurrent_dropout = recurrent_dropout,
                                     return_sequences=return_sequences)
            if bidirectional:
                layer = layers.Bidirectional(layer)

            model.add(layers.BatchNormalization())
            return layer

        for l in enumerate(range(num_layers -1)):
            model.add(create_layer())

        model.add(create_layer(return_sequences=False))
        model.add(layers.Dense(len(label_index), activation='softmax'))

        model.compile(optimizer=optimizers.Adam(), 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model_gru = model_gru = build_model_rnn(128,256, bidirectional=True)
model_gru.summary()

history_gru = model_gru.fit(x=x_train_pad, y=y_train_vect,  validation_data=(x_val_pad, y_val_vect),
                   epochs=10, batch_size=256)

chart_loss(history_dropout)
chart_loss(history_cnn)
chart_loss(history_gru)

chart_acc(history_dropout)
chart_acc(history_cnn)
chart_acc(history_gru)

def compare_loss(histories):
    epochs = range(1, len(list(histories.values())[0].history['val_loss']) + 1)

    for i, history in histories.items():
        val_loss = history.history['val_loss']
        plt.plot(epochs, val_loss, label='Validation Loss {}'.format(i))
            
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

compare_loss({'Dropout': history_dropout, 'CNN': history_cnn, 'GRU': history_gru})

def compare_accuracy(histories):
    epochs = range(1, len(list(histories.values())[0].history['val_acc']) + 1)

    for i, history in histories.items():
        val_loss = history.history['val_acc']
        plt.plot(epochs, val_loss, label='Validation Accuracy {}'.format(i))
            
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

compare_accuracy({'Dropout': history_dropout, 'CNN': history_cnn, 'GRU': history_gru})

def min_loss(history):
    val_loss = history.history['val_loss'] 
    return np.argmin(val_loss) + 1 

def accuracy(history, epoch):
    val_acc = history.history['val_acc']
    return val_acc[epoch - 1]

print('min loss for model dropout is {}'.format(min_loss(history_dropout)))
print('min loss for model cnn is {}'.format(min_loss(history_cnn)))
print('min loss for model gru is {}'.format(min_loss(history_gru)))

print('best validation accuracy for model dropout is {}'.format(
    accuracy(history_dropout, min_loss(history_dropout))))
print('best validation accuracy for model cnn is {}'.format(
    accuracy(history_cnn, min_loss(history_cnn))))
print('best validation accuracy for model gru is {}'.format(
    accuracy(history_gru, min_loss(history_gru))))

model = build_model_rnn(128,256, bidirectional=True)
model.summary()

history = model.fit(x=np.concatenate([x_train_pad, x_val_pad]), y=np.concatenate([y_train_vect, y_val_vect]), 
                    epochs=3, batch_size=256)

loss, acc = model.evaluate(x_test_pad, y_test_vect)

print('loss {}'.format(loss))
print('acc {}'.format(acc))

