# Import necessary libraries
import keras
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.layers import Convolution1D, LSTM
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
import itertools
import numpy as np
from keras.utils.np_utils import to_categorical

train = ["What would it cost to travel to the city on Monday?",
         "Need to travel this afternoon",
         "I want to buy a ticket",
         "Can I order a trip?", 
         "I would like to buy a ticket to Brussels", 
 
         "What will be the weather tomorrow?",
         "Will it rain this afternoon?",
         "The sunshine feels great",
         "Can you predict rain?",
         "Guess I should wear a jacket hey!",
 
        "Dit is geheel iets anders",
         "Kan ik dit goed vinden",
         "Wat is dit soms goed",
        "Maar anders is soms goed"]
 
T = "Buy a train ticket"
W = "Asking about the weather"
F = "Babble in 't Vlaamsch"

labelsTrain = [T,
               T,
               T,
               T,
               T,
 
               W,
               W,
               W,
               W,
               W,
 
               F,
               F,
               F,
               F]
 
test = [
        "Do you think it will be sunny tomorrow?",
        "What a wonderful feeling in the sun!",
        "How can I travel to Leuven?",
        "Can I buy it from you?",
        "Anders is heel goed"
       ]
labelsTest = [W, W, T, T, F]
 

tokenizer = Tokenizer()
all_texts = train + test # Combines two lists
tokenizer.fit_on_texts(all_texts) 
print(tokenizer.word_index)
print("\nLength of all text:",len(all_texts))

X_train = tokenizer.texts_to_matrix(train) # Convert the sentences directly to equal size arrays
X_test = tokenizer.texts_to_matrix(test)   # Convert the sentences directly to equal size arrays

print(train)
print("\nShape of X_train:",X_train.shape)
print("Shpae of X_test:",X_test.shape)

# Displaying first 2 rows of X_train:
for i in range(0,2):
    print(X_train[i])

all_labels = labelsTest + labelsTrain
labels = set(all_labels) # set -> function avoid duplicates
idx2labels = list(labels)
print(idx2labels)

# The enumerate() function adds a counter to an iterable.
label2idx = dict((v, i) for i, v in enumerate(labels)) # dictionary holding key and value
print(label2idx)

y_train = to_categorical([label2idx[w] for w in labelsTrain])
y_test = to_categorical([label2idx[w] for w in labelsTest])

print(y_test)

for i in labelsTest:
    print(label2idx[i],end=" ")

vocab_size = len(tokenizer.word_index) + 1
 
model = Sequential()
model.add(Embedding(2, 45, input_length= X_train.shape[1], dropout=0.2 ))
model.add(Flatten())
model.add(Dense(50, name='middle'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax', name='output')) 
 
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
 
model.fit(X_train, y=y_train, epochs=1500, verbose=0, validation_split=0.2, shuffle=True)
 
scores = model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

model.predict(tokenizer.texts_to_matrix(["Welke dag is het vandaag?"])).round()

embeddings_index = {}
# see here to download the pretrained model
# http://nlp.stanford.edu/projects/glove/
glove_data = './data/glove.6B/glove.6B.50d.txt'
f = open(glove_data,encoding='UTF-8')
for line in f:
    values = line.split()
    word = values[0]
    value = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = value
f.close()
 
print('Loaded %s word vectors.' % len(embeddings_index))
 

embedding_dimension = 50 # Setting embedding dimension to 50
word_index = tokenizer.word_index 
print(word_index)

embedding_matrix = np.zeros((len(word_index) + 1, embedding_dimension)) # Initiallizing the embedding matrix with 0

for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector[:embedding_dimension]
 

embedding_layer = Embedding(embedding_matrix.shape[0],
                            embedding_matrix.shape[1],
                            weights=[embedding_matrix],
                            input_length=len(word_index) + 1)

from keras.preprocessing.sequence import pad_sequences
X_train = tokenizer.texts_to_sequences(train) # Which Turns The input into numerical arrays 
print(X_train)

print("Length of word_index:",len(word_index))

X_train = pad_sequences(X_train, maxlen=len(word_index) + 1) 
# pad_sequence takes a LIST of sequences as an input (list of list) and will return a list of padded sequences.

# First 2 rows of X_train
for i in range(0,2):
    print(X_train[i])

model = Sequential()
model.add(embedding_layer)
model.add(Flatten())
model.add(Dense(50, activation='sigmoid'))
model.layers[0].trainable=False # bug in Keras or Theano
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax', name='output')) 
 
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
 
model.fit(X_train, y=y_train, epochs=2500, verbose=0, validation_split=0.2, shuffle=True)
 
scores = model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

model = Sequential()
model.add(embedding_layer)
model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2)) 
model.add(Dense(3, activation='softmax', name='output')) 
 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
 
model.fit(X_train, y=y_train, nb_epoch=1000, verbose=0, validation_split=0.2, shuffle=True)
 
scores = model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
 

