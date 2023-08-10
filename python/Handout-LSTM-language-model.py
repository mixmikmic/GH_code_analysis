import numpy as np
import urllib
from matplotlib import pyplot as plt
import random
import sys

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.layers.wrappers import TimeDistributed


path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
text = open(path).read().lower()
chars = sorted(list(set(text)))
vocab_size = len(chars)
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
print("Total number of chars:", len(text))
print("Vocabulary size:", vocab_size)

print(text[31000:31500])

maxlen = 40
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, vocab_size), return_sequences=False, name="lstm_1"))
model.add(Dense(vocab_size, name="dense_1"))
model.add(Activation('softmax', name="activation_1"))
model.summary(70)

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

print('Shape X', X.shape)
print('Shape y', y.shape)

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer = optimizer)

model.fit(X[:1000,:,:], y[:1000,:], batch_size=128, epochs=1)

maxlen = 40
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, vocab_size), return_sequences=True, name="lstm_1"))
model.add(TimeDistributed(Dense(vocab_size), name="dense_1"))#Check names to see how to load weights
model.add(Activation('softmax', name="activation_1"))
model.summary(70)

h5file = 'lstm-pretrained-weights.hdf5'
optimizer = RMSprop(lr=0.01)
model.load_weights(h5file)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

def log_likelihood(model, text):
    probs = model.predict(parse_text(text, vocab_size, padding=True)).squeeze()
    return sum([np.log(probs[i, char_indices[c]]) 
                 for i,c in enumerate(text[1:]) ]) 

def parse_text(text, vocab_size, padding=False):
    if padding:
        X = np.zeros((1, maxlen, vocab_size), dtype=np.bool)
    else:
        X = np.zeros((1, len(text), vocab_size), dtype=np.bool)
    for t, char in enumerate(text):
        X[0, t, char_indices[char]] = 1
    return X

print (log_likelihood(model, "the thing in itself"))

print (log_likelihood(model, "thethinginitself"))

print (log_likelihood(model, "the thingy in itself"))
print (log_likelihood(model, "itself thing the in"))

from itertools import permutations
from random import shuffle
char_list = list(u'ywh ')
perms = [''.join(perm) for perm in permutations(char_list)]
for p, t in sorted([(log_likelihood(model, text), text) for text in perms], reverse=True)[:5]:
    print(p, t)
print('-'*50)
for p, t in sorted([(log_likelihood(model, text), text) for text in perms], reverse=True)[-5:]:
    print(p, t)

from itertools import permutations
bow =  ['philosopher', 'kant', 'is', 'a']
perms = [' '+' '.join(perm)+' ' for perm in permutations(bow)]

for p, t in sorted([(log_likelihood(model, text), text) for text in perms], reverse = True)[:10]:
    print(p, t)

for p, t in sorted([(log_likelihood(model, text), text) for text in perms], reverse = True)[-10:]:
    print(p, t)

# Function to sample an index from a probability array:
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(diversity, model, sentence, n_chars, padding=True):
    print()
    print(sentence, end='')
    for i in range(n_chars):
        x = np.zeros((1, maxlen, vocab_size))
        if padding and len(sentence) < 40:
            space_array = [" "]*(40-len(sentence))
            for t, char in enumerate(space_array):
                x[0, t, char_indices[char]] = 1.
        for t, char in enumerate(sentence, 40-len(sentence)):
            x[0, t, char_indices[char]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds[-1], diversity)
        next_char = indices_char[next_index]
        sentence = sentence[1:] + next_char
        sys.stdout.write(next_char)
        sys.stdout.flush()
    return True

generate_text(0.5, model, 'the meaning of life is ', 400)



