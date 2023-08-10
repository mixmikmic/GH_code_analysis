## Much borrowed from https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM, Dropout
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.models import load_model
import numpy as np
import random
import sys

text = open("startrekepisodes.txt").read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
vocabulary_size = len(chars)
print('total chars:', vocabulary_size)
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# How long is a title?
titles = text.split('\n')
lengths = np.array([len(n) for n in titles])
print("Max:", np.max(lengths))
print("Mean:", np.mean(lengths))
print("Median:", np.median(lengths))
print("Min:", np.min(lengths))

# hence choose 30 as seuence length to train on.

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 30
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

X = np.zeros((len(sentences), maxlen), dtype=int)
y = np.zeros((len(sentences), vocabulary_size), dtype=np.bool)

for i in range(len(sentences)):
    X[i] = np.array([char_indices[x] for x in sentences[i]])
    y[i, char_indices[next_chars[i]]] = True
print("Done preparing training corpus, shapes of sets are:")
print("X shape: " + str(X.shape))
print("y shape: " + str(y.shape))
print("Vocabulary of characters:", vocabulary_size)





layer_size = 256
dropout_rate = 0.5
# build the model: a single LSTM
print('Build model...')
model_train = Sequential()
model_train.add(Embedding(vocabulary_size, layer_size, input_length=maxlen))

# LSTM part
model_train.add(Dropout(dropout_rate))
model_train.add(LSTM(layer_size, return_sequences=True))
model_train.add(Dropout(dropout_rate))
model_train.add(LSTM(layer_size))

# Project back to vocabulary
model_train.add(Dense(vocabulary_size))
model_train.add(Activation('softmax'))
model_train.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01))
model_train.summary()


# Training the Model.
model_train.fit(X, y, batch_size=64, epochs=10)

# model_train.fit(X, y, batch_size=64, epochs=20)

# Save model if necessary
model_train.save("keras-startrek-LSTM-model.h5")

# Load model if necessary.
model_train = load_model("keras-startrek-LSTM-model.h5")

# Build a decoding model (input length 1, batch size 1, stateful)
layer_size = 256
dropout_rate = 0.5

model_dec = Sequential()
model_dec.add(Embedding(vocabulary_size, layer_size, input_length=1, batch_input_shape=(1,1)))

# LSTM part
model_dec.add(Dropout(dropout_rate))
model_dec.add(LSTM(layer_size, stateful=True, return_sequences=True))
model_dec.add(Dropout(dropout_rate))
model_dec.add(LSTM(layer_size, stateful=True))

# project back to vocabulary
model_dec.add(Dense(vocabulary_size))
model_dec.add(Activation('softmax'))
model_dec.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01))
model_dec.summary()

# set weights from training model
model_dec.set_weights(model_train.get_weights())

## Sampling function

def sample_model(seed, model_name, length=400):
    '''Samples a charRNN given a seed sequence.'''
    generated = ''
    sentence = seed.lower()[:]
    generated += sentence
    print("Seed: ", generated)
    
    for i in range(length):
        x = np.array([char_indices[n] for n in sentence])
        x = np.reshape(x,(1,1))
        preds = model_name.predict(x, verbose=0)[0]
        next_index = sample(preds, 0.5)
        next_char = indices_char[next_index]
        
        generated += next_char
        sentence = sentence[1:] + next_char
    print("Generated: ", generated)

# Sample 1000 characters from the model using a random seed from the vocabulary.
sample_model(indices_char[random.randint(0,vocabulary_size-1)], model_dec, 1000)

def model_maker(model, layer_size=64, dropout_rate=0.5, num_layers=1, vocab_size=20, input_length=1, lr=0.01, train_mode=True):
    """Builds a charRNN model with variable layer size, number of layers, droupout, learning rate, and a training mode."""
    if train_mode:
        stateful = False
        input_shape = (None, input_length)
    else:
        stateful = True
        input_shape = (1, input_length)
    
    # Input embedding
    model.add(Embedding(vocab_size, layer_size, input_length=input_length, batch_input_shape=input_shape))
              
    # LSTM layers + 1
    for i in range(num_layers - 1):
        model.add(Dropout(dropout_rate))
        model.add(LSTM(layer_size, return_sequences=True, stateful=stateful))
    
    # Final LSTM layer
    model.add(Dropout(dropout_rate))
    model.add(LSTM(layer_size, stateful=stateful))

    # Project back to vocabulary
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=lr))
    model.summary()

# m = Sequential()
# model_maker(m, layer_size=128, vocab_size=vocabulary_size, input_length=30, train_mode=True)
# m.fit(X, y, batch_size=64, epochs=5)

