from theano.sandbox import cuda

get_ipython().magic('matplotlib inline')
#import utils_modified; reload(utils_modified)
#from utils_modified import *
from __future__ import division, print_function

import numpy as np
import random
import sys

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Embedding, Activation, LSTM, Flatten, Dropout, Lambda
from keras.models import Model, Sequential
#from keras.layers import merge # deprecated in Keras 2
from keras.layers.merge import Add, add
#from keras.engine.topology import Merge # deprecated in Keras 2
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers.convolutional import *
from keras.utils.data_utils import get_file
from keras.utils.np_utils import to_categorical

# https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
text = open(path).read()#.lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

lag = 6

idx = [char_indices[c] for c in text]

Xs = []
for l in range(lag):
    cdat = [idx[i+l] for i in xrange(0, len(idx)-1-lag, lag)]
    X = np.stack(cdat[:-2])
    Xs.append(X)

cdat = [idx[i+(l+1)] for i in xrange(0, len(idx)-1-lag, lag)]
Y = np.stack(cdat[:-2])

len(Xs), Xs[0].shape

def show_top_next(mdl, inp):
    pad_inp = (' '*lag)+inp
    idxs = [char_indices[c] for c in pad_inp][-lag:]
    ps = mdl.predict([np.array([i]) for i in idxs])
    probas = ps[0]/np.sum(ps)
    for i in range(5):
        index = np.random.choice(range(len(chars)), size=None, replace=True, p=probas)
        print(inp+chars[index])
    print('')
    
teststrs = ['this i','hersel','himsel','moral','moralit','knowledg','logica','hypothesi']

hdim = 250

def Maker1():
    inputs = [Input(shape=(1,), dtype='int64') for i in range(lag)]

    E = Embedding(output_dim=hdim, input_dim=len(chars), input_length=1)

    Di2h = Dense(hdim, activation='relu')
    Dh2h = Dense(hdim, activation='relu', kernel_initializer='identity')

    #hidden = ... CONSTANT ZERO TENSOR IN KERAS ?

    e = Di2h(Flatten()(E(inputs[0])))
    hidden = e
    for i in range(1,lag):
        e = Di2h(Flatten()(E(inputs[i])))
        #hidden = merge([e, Dh2h(hidden)], mode='sum')
        hidden = add([e, Dh2h(hidden)])
    predictions = Dense(len(chars), activation='softmax')(hidden)

    model = Model(inputs=inputs, outputs=predictions)

    model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

model1 = Maker1()

model1.optimizer.lr = 1e-3
hist1 = model1.fit(Xs, to_categorical(Y,len(chars)), batch_size=100, epochs=25)

for teststr in teststrs:
    show_top_next(model1, teststr)

def Maker2():
    inputs = [Input(shape=(1,), dtype='int64') for i in range(lag)]

    E = Embedding(output_dim=hdim, input_dim=len(chars), input_length=1)

    Di2h = Dense(hdim, activation='relu')
    Dh2h = Dense(hdim, activation='relu', kernel_initializer='identity')

    #hidden = ... CONSTANT ZERO TENSOR IN KERAS ?
    
    e = Di2h(Flatten()(E(inputs[0])))
    hidden = BatchNormalization()(e)
    for i in range(1,lag):
        e = Di2h(Flatten()(E(inputs[i])))
        #hidden = merge([BatchNormalization()(e), Dh2h(hidden)], mode='sum') # deprecated
        hidden = add([BatchNormalization()(e), Dh2h(hidden)])
    predictions = Dense(len(chars), activation='softmax')(hidden)

    model = Model(inputs=inputs, outputs=predictions)

    model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

model2 = Maker2()

model2.optimizer.lr = 1e-3
hist2 = model2.fit(Xs, to_categorical(Y,len(chars)), batch_size=100, epochs=25)

for teststr in teststrs:
    show_top_next(model2, teststr)

import matplotlib.pyplot as plt
plt.figure(figsize=(10,7))
plt.plot(hist1.history['loss'])
plt.plot(hist2.history['loss'])
#plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['without BN','with BN'], loc='upper right')
plt.show()



