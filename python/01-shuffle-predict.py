import re
import numpy as np
import random

from collections.abc import Sequence

from cached_property import cached_property
from gensim.models import KeyedVectors

from sklearn.model_selection import train_test_split

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Embedding, Dropout

w2v = KeyedVectors.load_word2vec_format(
    '../data/GoogleNews-vectors-negative300.bin.gz',
    binary=True,
)

def tokenize(text):
    return re.findall('\w+', text)

def sent_embed_iter(text):
    for token in tokenize(text):
        if token in w2v:
            yield w2v[token]

def plot_embed_iter(sents):
    for sent in sents:
        yield from sent_embed_iter(sent)

def parse_plots(path):
    """Generate plot sentences.
    """
    with open(path) as fh:
        
        plot = []
        for line in fh.read().splitlines():
            
            if line != '<EOS>':
                plot.append(line)
                
            else:
                yield plot
                plot = []

plots = list(parse_plots('../data/plots/plots'))

x, y = [], []

for plot in plots[:1000]:
    
    x.append(list(plot_embed_iter(plot)))
    y.append(True)
    
    shuffled = random.sample(plot, len(plot))
    
    x.append(list(plot_embed_iter(shuffled)))
    y.append(False)

x = pad_sequences(x, 1000, padding='post', dtype=float)

x.shape

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

x_train.shape

x_test.shape

model = Sequential()
model.add(LSTM(128, input_shape=x_train[0].shape, return_sequences=True))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'],
)

model.fit(x_train, y_train)

model.evaluate(x_test, y_test)

model.metrics_names



