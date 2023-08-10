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
import math

import random

from dataset import atti_dirigenti

# to make the experimens replicable
random.seed(123456)

(x_train, y_train), (x_val, y_val), (x_test, y_test) = atti_dirigenti.load_data(num_words=10000, remove_stopwords=True)

label_index = atti_dirigenti.get_labels()
len(label_index)

from keras.preprocessing import sequence

print('max length of objects {}'.format(max(map(len, x_train))))

maxlen = 100

x_train_seq = sequence.pad_sequences(x_train, maxlen=maxlen)
x_val_seq = sequence.pad_sequences(x_val, maxlen=maxlen)
x_test_seq = sequence.pad_sequences(x_test, maxlen=maxlen)

x_train_seq.shape

def vectorize_sequences(sequences, dimension=10000):
    data = np.zeros((len(sequences), dimension), dtype=np.float32)
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

y_train_vect

def build_model_only_embeddings(embed_size):
    model = models.Sequential()
    model.add(layers.Embedding(input_dim=dimension, output_dim=embed_size, input_length=maxlen, name='embed'))
    model.add(layers.Flatten())
    model.add(layers.Dense(len(label_index), activation='softmax', name='softmax'))
    model.compile(optimizer=optimizers.Adam(), 
          loss='categorical_crossentropy',
          metrics=['accuracy'])
    return model

model_only_embed_small = build_model_only_embeddings(32)
history_embed_small = model_only_embed_small.fit(x=x_train_seq, y=y_train_vect, validation_data=(x_val_seq, y_val_vect),
                   epochs=10, batch_size=256)

model_only_embed_medium = build_model_only_embeddings(64)
history_embed_medium = model_only_embed_medium.fit(x=x_train_seq, y=y_train_vect, validation_data=(x_val_seq, y_val_vect),
                   epochs=10, batch_size=256)

model_only_embed_large = build_model_only_embeddings(128)
history_embed_large = model_only_embed_large.fit(x=x_train_seq, y=y_train_vect, validation_data=(x_val_seq, y_val_vect),
                   epochs=10, batch_size=256)

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
    plt.yticks(np.arange(0,2.2, step=0.2))
    plt.xticks(epochs)
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
    plt.yticks(np.arange(0.4,1.05, step=0.05))
    plt.xticks(epochs)
    plt.legend()
    plt.show()

chart_loss(history_embed_small)
chart_loss(history_embed_medium)
chart_loss(history_embed_large)

chart_acc(history_embed_small)
chart_acc(history_embed_medium)
chart_acc(history_embed_large)

def min_loss(history):
    val_loss = history.history['val_loss'] 
    return np.argmin(val_loss) + 1 

def accuracy(history, epoch):
    val_acc = history.history['val_acc']
    return val_acc[epoch - 1]

print('min loss for model small is {}'.format(min_loss(history_embed_small)))
print('min loss for model medium is {}'.format(min_loss(history_embed_medium)))
print('min loss for model large is {}'.format(min_loss(history_embed_large)))

print('best validation accuracy for model small is {}'.format(
    accuracy(history_embed_small, min_loss(history_embed_small))))
print('best validation accuracy for model medium is {}'.format(
    accuracy(history_embed_medium, min_loss(history_embed_medium))))
print('best validation accuracy for model large is {}'.format(
    accuracy(history_embed_large, min_loss(history_embed_large))))

def build_model_dropout(neurons):
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

def build_model_embeddings(neurons, embed_size):
    model = models.Sequential()
    model.add(layers.Embedding(input_dim=dimension, output_dim=embed_size, input_length=maxlen))
    model.add(layers.Flatten())
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
                   epochs=10, batch_size=256)

model_embed = build_model_embeddings(256,128)

model_embed.summary()

history_embed = model_embed.fit(x=x_train_seq, y=y_train_vect, validation_data=(x_val_seq, y_val_vect),
                   epochs=10, batch_size=256)

chart_loss(history_dropout)
chart_loss(history_embed)

chart_acc(history_embed)
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

compare_loss({'embedding': history_embed, 'Dropout': history_dropout})

def compare_accuracy(histories):
    epochs = range(1, len(list(histories.values())[0].history['val_acc']) + 1)

    for i, history in histories.items():
        val_loss = history.history['val_acc']
        plt.plot(epochs, val_loss, label='Validation Accuracy {}'.format(i))
            
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

compare_accuracy({'embedding': history_embed, 'Dropout': history_dropout})

print('min loss for model Embed {}'.format(min_loss(history_embed)))
print('min loss for model Dropout is {}'.format(min_loss(history_dropout)))

print('best validation accuracy for model Embed {}'.format(
    accuracy(history_embed, min_loss(history_embed))))
print('best validation accuracy for model Dropout {}'.format(
    accuracy(history_dropout, min_loss(history_dropout))))

model = build_model_embeddings(256,128)

history = model.fit(x=np.concatenate([x_train_seq, x_val_seq]), y=np.concatenate([y_train_vect, y_val_vect]), epochs=4, batch_size=256)

loss, acc = model.evaluate(x_test_seq, y_test_vect)

print('loss {}'.format(loss))
print('acc {}'.format(acc))

