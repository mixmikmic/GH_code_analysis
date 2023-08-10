import pandas as pd

reviews = pd.read_csv('../data/en_reviews.csv', sep='\t', header=None, names =['rating', 'text'])
reviews = reviews['text'].tolist()
print(reviews[:2])

data = '\n'.join(map(lambda x: x.replace('\n', ' '), reviews))

chars = sorted(list(set(data)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

print(len(chars))

MAXLEN = 40
STEP = 20

sequences = []
next_chars = []
for i in range(0, len(data) - MAXLEN, STEP):
    sequences.append(data[i: i + MAXLEN])
    next_chars.append(data[i + MAXLEN])
    

import numpy as np

X_train = np.zeros((len(sequences), MAXLEN, len(chars)), dtype=np.bool)
y_train = np.zeros((len(sequences), len(chars)), dtype=np.bool)

for i, sequences in enumerate(sequences):
    for t, char in enumerate(sequences):
        X_train[i, t, char_indices[char]] = 1
        y_train[i, char_indices[next_chars[i]]] = 1

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import RMSprop

model = Sequential()
model.add(LSTM(128, input_shape=(MAXLEN, len(chars)), dropout=0.5))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate(seed, temperature=1.0):
    sentence = MAXLEN * '\n' + seed
    sentence = sentence[-MAXLEN:]
    generated = seed

    next_char = None
    while next_char != '\n':
        X_pred = np.zeros((1, MAXLEN, len(chars)))
        for t, char in enumerate(sentence):
            X_pred[0, t, char_indices[char]] = 1.

        y_pred = model.predict(X_pred, verbose=0)[0]
        next_index = sample(y_pred, temperature)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char
    return generated[0:-1]

EPOCHS = 20

old_loss = None
for iteration in range(1, EPOCHS + 1):
    print()
    print('-' * 50)
    print('Iteration', iteration)
            
    history = model.fit(X_train, y_train, batch_size=128, epochs=1)
    loss = history.history['loss'][0]
    if old_loss != None and old_loss < loss:
        print("Loss explosion.")
        break
    old_loss = loss
    start_index = np.random.randint(0, len(data) - MAXLEN - 1)
    sentence = data[start_index: start_index + MAXLEN]
    print(generate(sentence))

