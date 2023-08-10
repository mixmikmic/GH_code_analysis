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

import math

import random

# to make the experimens replicable
random.seed(123456)

dataset_path = '../data/dataset-dirigenti-split.pkl'

with open(dataset_path, 'rb') as f:
    train_samples, train_labels, val_samples, val_labels, test_samples, test_labels = pickle.load(f)

samples = np.concatenate([train_samples,val_samples, test_samples])
samples.shape

labels = np.concatenate([train_labels, val_labels, test_labels])
labels.shape

max_features = 10000

from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(samples)

with open('id_word.tsv','w') as f:
    for word, ids in tokenizer.word_index.items():
        f.write('{}\t{}\n'.format(ids,word))

train_sequences = tokenizer.texts_to_sequences(train_samples)
val_sequences = tokenizer.texts_to_sequences(val_samples)
test_sequences = tokenizer.texts_to_sequences(test_samples)

for s in test_sequences[:2]:
    print(s) 

index_to_labels = dict(enumerate(set(labels)))
labels_to_index = {v:k for k,v in index_to_labels.items()}

list(labels_to_index.items())[:5]

len(labels_to_index)

encoded_train_labels = [labels_to_index[l] for l in train_labels]
encoded_val_labels = [labels_to_index[l] for l in val_labels]
encoded_test_labels = [labels_to_index[l] for l in test_labels]

encoded_train_labels[:10]

from keras.preprocessing import sequence

print('max length of objects {}'.format(max(map(len, train_sequences))))

maxlen = 150

x_train_pad = sequence.pad_sequences(train_sequences, maxlen=maxlen)
x_val_pad = sequence.pad_sequences(val_sequences, maxlen=maxlen)
x_test_pad = sequence.pad_sequences(test_sequences, maxlen=maxlen)

x_train_pad.shape

def vectorize_sequences(sequences, dimension=10000):
    data = np.zeros((len(sequences), dimension), dtype=np.float16)
    for i, sequence in enumerate(sequences):
        data[i,sequence] = 1.
    return data

x_train = vectorize_sequences(train_sequences, max_features)
x_val = vectorize_sequences(val_sequences, max_features)
x_test = vectorize_sequences(test_sequences, max_features)

def to_one_hot(labels):
    results = np.zeros((len(labels), len(set(labels))), dtype=np.float32)
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

y_train = to_one_hot(encoded_train_labels)
y_val = to_one_hot(encoded_val_labels)
y_test = to_one_hot(encoded_test_labels)

y_train.shape

import matplotlib.pyplot as plt

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

def build_model_dropout(neurons):
    model = models.Sequential()
    model.add(layers.Dense(neurons, activation='relu', input_shape=(x_train.shape[-1], )))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(neurons, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(neurons, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(len(labels_to_index), activation='softmax'))
    
    model.compile(optimizer=optimizers.Adam(), 
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model

def build_model_cnn(embed_size):
    model = models.Sequential()
    model.add(layers.Embedding(input_dim=max_features, output_dim=embed_size, input_length=maxlen, name='embed'))
    model.add(layers.Conv1D(16, 5, activation='relu'))
    model.add(layers.MaxPooling1D(5))
    model.add(layers.Conv1D(16, 5, activation='relu'))
    model.add(layers.Bidirectional(layers.GRU(64, activation='relu',
                         dropout= 0.5,
                         recurrent_dropout = 0.1,
                         return_sequences=False)))
    model.add(layers.Dense(len(labels_to_index), activation='softmax'))
    
    model.compile(optimizer=optimizers.Adam(), 
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    return model

def build_model_lstm(neurons, embed_size):
    model = models.Sequential()
    model.add(layers.Embedding(input_dim=max_features, output_dim=embed_size, input_length=maxlen, name='embed'))
    model.add(
#         layers.Bidirectional(
        layers.GRU(neurons, activation='relu',
                         dropout= 0.1,
                         recurrent_dropout = 0.2,
                         return_sequences=True))
#              )
#     model.add(layers.BatchNormalization())
    model.add(layers.GRU(neurons, activation='relu',
                     dropout= 0.1,
                     recurrent_dropout = 0.2,
                     return_sequences=False))
#     model.add(layers.BatchNormalization())
    model.add(layers.Dense(len(labels_to_index), activation='softmax'))
    
    model.compile(optimizer=optimizers.Adam(), 
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model

model_dropout = build_model_dropout(256)

history_dropout = model_dropout.fit(x=x_train, y=y_train, validation_data=(x_val, y_val),
                   epochs=20, batch_size=256, callbacks=callbacks)

get_ipython().run_line_magic('xdel', 'model_dropout')

import gc
gc.collect()

model_cnn = build_model_cnn(64)

from keras.callbacks import ReduceLROnPlateau

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, mode='min',cooldown=0, min_lr=0)
]

history_cnn = model_cnn.fit(x=x_train_pad, y=y_train,  validation_data=(x_val_pad, y_val),
                   epochs=20, batch_size=256)

get_ipython().run_line_magic('xdel', 'model_cnn')

model_lstm = build_model_lstm(64,64)
model_lstm.summary()

history_lstm = model_lstm.fit(x=x_train_pad, y=y_train,  validation_data=(x_val_pad, y_val),
                   epochs=20, batch_size=256, callbacks=callbacks)

get_ipython().run_line_magic('xdel', 'model_lstm')

chart_loss(history_lstm)
chart_loss(history_dropout)

chart_acc(history_lstm)
chart_acc(history_dropout)

def compare_loss(histories):
    epochs = range(1, len(list(histories.values())[0].history['val_loss']) + 1)

    for i, history in histories.items():
        val_loss = history.history['val_loss']
        plt.plot(epochs, val_loss, label='Validation Loss {}'.format(i))
            
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

compare_loss({'LSTM': history_lstm, 'Dropout': history_dropout})

def compare_accuracy(histories):
    epochs = range(1, len(list(histories.values())[0].history['val_acc']) + 1)

    for i, history in histories.items():
        val_loss = history.history['val_acc']
        plt.plot(epochs, val_loss, label='Validation Accuracy {}'.format(i))
            
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

compare_accuracy({'LSTM': history_lstm, 'Dropout': history_dropout})

model = build_model_lstm(64,64)

history = model.fit(x=np.concatenate([x_train_pad, x_val_pad]), y=np.concatenate([y_train, y_val]), 
                    epochs=2, batch_size=256)

loss, acc = model.evaluate(x_test_pad, y_test)

print('loss {}'.format(loss))
print('acc {}'.format(acc))



