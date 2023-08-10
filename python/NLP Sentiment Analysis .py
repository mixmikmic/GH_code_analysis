import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from scipy.spatial.distance import cdist

# from tf.keras.models import Sequential  # This does not work!
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import save_model
import csv 
import pandas as pd 
from IPython.display import display, HTML
import h5py
from random import shuffle

def load_dataset(file = 'sentiment.txt'):
    with open(file, 'r') as f:
        labels = []
        text = []

        lines = f.readlines()
    shuffle(lines)
    for line in lines:
        data = line.split('\t')
        if len(data) == 2:
            labels.append(data[0])
            text.append(data[1].rstrip())
    return text,labels
    
x_train_text , y_train = load_dataset()

data_text = x_train_text
idx = 5
print(x_train_text[idx],'\n',y_train[idx])

import re
def process(txt):
    out = re.sub(r'[^a-zA-Z0-9\s]', '', txt)
    out = out.split()
    out = [word.lower() for word in out]
    return out

def tokenize(thresh = 5):
    count  = dict()
    idx = 1
    word_index = dict()
    for txt in data_text:
        words = process(txt)
        for word in words:
            if word in count.keys():
                count[word] += 1
            else:
                count[word]  = 1
    most_counts = [word for word in count.keys() if count[word]>=thresh]
    for word in most_counts:
        word_index[word] = idx
        idx+=1
    return word_index

num_words = None
word_index = tokenize()
num_words = len(word_index)
print('length of the dictionary ',len(word_index))

def getMax(data):
    max_tokens = 0 
    for txt in data:
        if max_tokens < len(txt.split()):
            max_tokens = len(txt.split())
    return max_tokens
max_tokens = getMax(x_train_text)

def create_sequences(data):
    tokens = []
    for txt in data:
        words = process(txt)
        seq = [0] * max_tokens
        i = 0 
        for word in words:
            start = max_tokens-len(words)
            if word.lower() in word_index.keys():
                seq[i+start] = word_index[word]
            i+=1
        tokens.append(seq)        
    return np.array(tokens)
            
    

print(create_sequences(['awesome movie']))

x_train_tokens = create_sequences(x_train_text)


model = Sequential()
embedding_size = 8
model.add(Embedding(input_dim=num_words,
                    output_dim=embedding_size,
                    input_length=max_tokens,
                    name='layer_embedding'))

model.add(GRU(units=16, name = "gru_1",return_sequences=True))
model.add(GRU(units=8, name = "gru_2" ,return_sequences=True))
model.add(GRU(units=4, name= "gru_3"))
model.add(Dense(1, activation='sigmoid',name="dense_1"))
optimizer = Adam(lr=1e-3)
model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

model.fit(x_train_tokens, y_train,
          validation_split=0.05, epochs=5, batch_size=32)

txt = ["awesome movie","Terrible movie","that movie really sucks","I like that movie"]
print(create_sequences(txt)[0])
pred = model.predict(create_sequences(txt))
print('\n prediction for \n',pred[:,0])

save_model(
    model,
    "keras.h5",
    overwrite=True,
    include_optimizer=True
)

model.summary()

def create_csv(file):
    with open(file, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for key in word_index.keys():
            writer.writerow([key,word_index[key]])

create_csv('dict.csv')

