import os
os.environ['KERAS_BACKEND'] = 'theano'
from keras.models import Sequential
from keras.layers import Dense, Embedding, Activation, Dropout
from keras.layers import GRU
from keras.regularizers import l2, activity_l2
from keras.optimizers import RMSprop
from IPython.display import clear_output
import numpy as np
import math
from datetime import datetime
import time
import random
import sys
import string
import h5py

np.random.seed()
random.seed()

mask = '_'
chars = None
with open('./domains.txt') as handle:
    text = handle.read().lower()[:-1]
    chars = sorted(list(set(text + mask)))
    print 'corpus length:', len(text)
    
weights_path = './model.hdf5'
print 'total chars:', len(chars)
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

def pad_name(name, max_length, skip = 0):
    padded_name = mask * (max_length - len(name) - skip)
    padded_name += name
    
    return padded_name

maxlen = 3
domains = []
next_chars = []
maxlen2 = 0
with open('./domains.txt', 'r') as handle:
    for line in handle:
        line = line.strip()
        maxlen2 = max(maxlen2, len(line))

step = 1
batch_size = maxlen2 * 100
with open('./domains.txt', 'r') as handle:
    padded_names = ""
    for line in handle:
        line = line.strip()                    
        padded_names += pad_name(line, maxlen2)

for i in range(0, len(padded_names) - maxlen, step):
    domains.append(padded_names[i: i + maxlen])
    next_chars.append(padded_names[i + maxlen])

for i in range(0, maxlen2*2):
    print "%s -> %s" % (domains[i], next_chars[i])

print 'nb sequences:', len(domains), len(next_chars)
print "batch size: %d maxlen2: %d" % (batch_size, maxlen2)

X = np.zeros((len(domains), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(domains), len(chars)), dtype=np.bool)
for i, domain in enumerate(domains):
    for t, char in enumerate(domain):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

model = Sequential()
model.add(GRU(128, dropout_W=0.015, dropout_U=0.015, input_shape=(maxlen, len(chars)), return_sequences=False, stateful=False))

#for a hidden layer, uncomment this one
#model.add(GRU(128, dropout_W=0.015, dropout_U=0.015, return_sequences=False, stateful=False))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

if os.path.exists(weights_path):
    print "loading existing model.."
    model.load_weights(weights_path)

def generate_alphabet_names():
    seeds = string.ascii_lowercase
    generate_names_amount = len(seeds)
    diversity = random.uniform(0.05, 0.5)

    print "Name AI by Peter Willemsen <peter@codebuffet.co>\nCreating %d names with diversity %f" % (generate_names_amount, diversity)
    for i in range(0, generate_names_amount):
        seed = pad_name(seeds[i], maxlen)
        sentence = seed
        generated = seed
        domains = generated

        for i in range(maxlen2 * 1):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            domains += next_char
        print domains.replace("_", "")

iteration = 0
while True:
    print 'Iteration', iteration
    model.fit(X, y, batch_size=batch_size, shuffle=False, nb_epoch=1, verbose=0)
    model.save_weights(weights_path, overwrite=True)
    clear_output()    
    generate_alphabet_names()      
        
    iteration += 1

generate_alphabet_names()



