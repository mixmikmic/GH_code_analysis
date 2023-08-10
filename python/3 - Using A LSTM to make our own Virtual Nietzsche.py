from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop, Adam
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys

# Read the entire file containing nietzsche's works
path = './data/nietzsche.txt'
text = open(path).read().lower()

# Output the length of the corpus
print('corpus length:', len(text))

# Create a sorted list of the characters
chars = sorted(list(set(text)))
print('total chars:', len(chars))

# Create a dictionary where given a character, you can look up the index and vice versa
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []

# Step through the text via 3 characters at a time, taking a sequence of 40 bytes at a time. 
# There will be lots ofo overlap
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen]) # range from current index i for max length characters 
    next_chars.append(text[i + maxlen]) # the next character after that 
print('Number of sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1
print('Finished created vectors')
print('Size of patterns:', len(X[0]))

# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

print("Compiling model complete...")


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

diversity = 0.5
print('Diversity: ', diversity)

# The training
print('Training...')
history = model.fit(X, y, batch_size=128, nb_epoch=20)

# Save the model
model.save_weights('nietzsche.weights')
json = model.to_json()
f = open('nietzsche.json','w')
f.write(json) 
f.close() 

# Save the model
model.load_weights('nietzsche.weights')

# Check out what our model predicts
sentence = 'those who will not inherit the kingdom o'
x = np.zeros((1, maxlen, len(chars)))
for t, char in enumerate(sentence):
    x[0, t, char_indices[char]] = 1.
    
print(model.predict(x, verbose=0)[0])
print(sum(model.predict(x, verbose=0)[0]))

generated = ''
original = sentence
# Predict the next 400 characters based on the seed
for i in range(400):
    x = np.zeros((1, maxlen, len(chars)))
    for t, char in enumerate(sentence):
        x[0, t, char_indices[char]] = 1.

    preds = model.predict(x, verbose=0)[0]
    next_index = sample(preds, diversity)
    next_char = indices_char[next_index]

    generated += next_char
    sentence = sentence[1:] + next_char

print(original + generated)



