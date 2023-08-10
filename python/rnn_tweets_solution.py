import numpy as np
import pandas as pd
import re

from keras.callbacks import LambdaCallback
from keras.layers import Dense, LSTM, Activation
from keras.models import Sequential
from keras.optimizers import RMSprop

df = pd.read_csv('trump_tweets.csv', delimiter=',', header=0)
df = df[pd.notnull(df['text'])]
text = [re.sub(r'https?:\/\/.*[\r\n]*', '', sent, flags=re.MULTILINE)
          .strip() for sent in df['text']][:5000]

df

char_to_index = {}
index_to_char = {}
start_token = 0
end_token = 1
num_chars = 2
for sentence in text:
    for char in sentence:
        if char not in char_to_index:
            char_to_index[char] = num_chars
            index_to_char[num_chars] = char
            num_chars += 1

input_len = 25
data = []
labels = []
for sentence in text:
    sent_list = [start_token] + [char_to_index[c] for c in sentence] + [end_token]
    sent_onehot = np.concatenate((np.zeros((input_len-1, num_chars)),
                                  np.eye(num_chars)[sent_list]), axis=0)
    for i in range(len(sent_list) - 1):
        data.append(sent_onehot[i:i+input_len])
        labels.append(sent_onehot[i+input_len])
data = np.stack(data, axis=0)
labels = np.stack(labels, axis=0)

hidden_neurons = 200
model = Sequential()
model.add(LSTM(hidden_neurons, input_shape=(input_len, num_chars)))
model.add(Dense(num_chars, activation='softmax'))
print(model.summary())

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_sentence():
    max_sent_length = 140
    end_sentence = False
    sent = np.zeros((input_len, num_chars))
    sent[-1, start_token] = 1
    
    generated = ''
    sent_len = 0
    while not end_sentence:
        sent_input = np.expand_dims(sent[-input_len:sent.shape[0]], axis=0)
        char_probs = model.predict(sent_input, verbose=0)
        next_char = sample(np.squeeze(char_probs, axis=0))
        if next_char == end_token or sent_len == max_sent_length:
            end_sentence = True
            print(generated)
        else:
            char_onehot = np.expand_dims(np.eye(num_chars)[next_char], axis=0)
            sent = np.concatenate((sent, char_onehot), axis=0)
            if not(next_char == 0 or next_char == 1):
                generated += index_to_char[next_char]
            sent_len += 1

for _ in range(5):
    generate_sentence()

def on_epoch_end(epoch, logs):
    print()
    print('----- Generating text after Epoch: %d' % epoch)
    for i in range(3):
        generate_sentence()
        print()

optimizer = RMSprop(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
model.fit(data, labels, batch_size=128, epochs=5,
          callbacks=[LambdaCallback(on_epoch_end=on_epoch_end)])



