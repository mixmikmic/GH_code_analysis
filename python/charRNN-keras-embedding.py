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

#path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
# text = open("tiny-shakespeare.txt").read().lower()
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

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 50
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
    X[i] = np.array(map((lambda x: char_indices[x]), sentences[i]))
    y[i, char_indices[next_chars[i]]] = True
print("Done converting y to one-hot.")
print("Done preparing training corpus, shapes of sets are:")
print("X shape: " + str(X.shape))
print("y shape: " + str(y.shape))
print("Vocabulary of characters:", vocabulary_size)

# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(Embedding(vocabulary_size, 128, input_length=maxlen))
model.add(LSTM(128))
model.add(Dense(vocabulary_size))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01))
model.summary()

# model.add(LSTM(128, return_sequences=True))
#model.add(Dropout(0.5))
#model.add(LSTM(128, return_sequences=True))
#model.add(Dropout(0.5))



# Training the Model.
model.fit(X, y, batch_size=128, epochs=20)

model.save("keras-shakespeare-LSTM-model-emb.h5")

model.save("keras-startrek-LSTM-model-emb.h5")





model_dec = Sequential()
model_dec.add(Embedding(vocabulary_size, 128, input_length=1, batch_input_shape=(1,1)))
model_dec.add(LSTM(128, stateful=True))
model_dec.add(Dense(vocabulary_size))
model_dec.add(Activation('softmax'))
model_dec.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01))
model_dec.summary()

model_train = load_model("keras-shakespeare-LSTM-model-emb.h5")
model_dec.set_weights(model_train.get_weights())

model = load_model("keras-shakespeare-LSTM-model-emb.h5")
quote = "Be not afraid of greatness: some are born great, some achieve greatness, and some have greatness thrust upon them."
quote = quote.lower()

def sample_model(seed, length=400):
    generated = ''
    sentence = seed.lower()[:50]
    generated += sentence
    print("Seed: ", generated)
    
    for i in range(length):
        x = np.array(map((lambda x: char_indices[x]), sentence))
        x = np.reshape(x,(1,50))
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, 0.5)
        next_char = indices_char[next_index]
        
        generated += next_char
        sentence = sentence[1:] + next_char
    print("Generated: ", generated)

sample_model(quote, 1000)

indices_char[random.randint(0,vocabulary_size-1)]


# Load up the prediction model
model_train = load_model("keras-startrek-LSTM-model-emb.h5")
model_dec.set_weights(model_train.get_weights())

# model_train = load_model("keras-startrek-LSTM-model-emb.h5")
# model_dec.set_weights(model_train.get_weights())

def sample_model(seed, model_name, length=400):
    generated = ''
    sentence = seed.lower()[:]
    generated += sentence
    print("Seed: ", generated)
    
    for i in range(length):
        x = np.array(map((lambda x: char_indices[x]), sentence))
        x = np.reshape(x,(1,1))
        preds = model_name.predict(x, verbose=0)[0]
        next_index = sample(preds, 0.5)
        next_char = indices_char[next_index]
        
        generated += next_char
        sentence = sentence[1:] + next_char
    print("Generated: ", generated)

sample_model(indices_char[random.randint(0,vocabulary_size-1)], model_dec, 1000)

