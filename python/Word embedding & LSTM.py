from keras.preprocessing.text import Tokenizer

sample_text = ["@user when a father is dysfunctional and is so selfish he drags his kids into his dysfunction. #run",
              "it's unbelievable that in the 21st century we'd need something like this. again. #neverump #xenophobia",
              "@user @user thanks for #lyft credit i can't use cause they don't offer wheelchair vans in pdx. #disapointed #getthanked"]
num_words = 100
tokenizer = Tokenizer(nb_words=num_words) ## Tokenize the texts

tokenizer.fit_on_texts(sample_text) ## Assiging inxed to all token

print tokenizer.word_index ## See the index of all tokens 
#print "#### #######"

#import enchant

print tokenizer.texts_to_sequences(["we'd need something like this. again. #neverump #xenophobia"])

#### Load the Pre Trained Glove Word Embedding 

from tqdm import tqdm
import numpy as np

embeddings_index = {}
f = open('glove.6B.300d.txt')
for line in tqdm(f):
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

#embeddings_index
import numpy as np
#print embeddings_index['car']## Find the word vector
print np.corrcoef(embeddings_index['car'],embeddings_index['bus'])

## Create the word Embedding Matrix
word_index = tokenizer.word_index  ### Number of words in the text sample
print len(word_index)
embedding_matrix = np.zeros((len(word_index) + 1, 300)) ### use same dimension length of embedding vector
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

print embedding_matrix.shape

temp = pd.DataFrame(embedding_matrix.tolist())
temp



## Create Embedding Layer
from keras.layers import Embedding
#embedding_layer = Embedding(len(word_index) + 1, 300, input_length=40, dropout=0.2)
embedding_layer = Embedding(embedding_matrix.shape[0],
                            embedding_matrix.shape[1],
                            weights=[embedding_matrix],
                            input_length=5)  ## Input Length is maximum words to be considered

## from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.sequence import pad_sequences
X = tokenizer.texts_to_sequences(sample_text)
X = pad_sequences(X, maxlen=5)
print(X)
#y = [0,1,0]



#from keras.layers import Bidirectional
from keras.models import Sequential
from keras.layers import Dense, Input, Flatten
from keras.layers.recurrent import LSTM, GRU

model = Sequential()
model.add(embedding_layer)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.layers[0].trainable=False
model.compile(loss='binary_crossentropy', optimizer='rmsprop')
model.fit(X, y=y, batch_size=20, nb_epoch=700, verbose=0, validation_split=0.2)

import pandas as pd
data_train = pd.read_csv('train_E6oV3lV.csv')
data_train.head()

#### Bulid the Final Model for Text classification

import numpy as np
import pandas as pd
import cPickle
from collections import defaultdict
import re

from bs4 import BeautifulSoup

import sys
import os

os.environ['KERAS_BACKEND']='theano'

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout,GlobalAveragePooling1D
from keras.models import Model

from nltk.corpus import stopwords
stops = set(stopwords.words("english"))

## Add spell checker and short form to long form 
remove_words = ['user',"frm", "u", "urs", "n", "ur", "b", "mro", "mo", "tmr", "k", "ok",
"lol", "haha", "w", "moro", "yah", "cya", "cu", "eh", "hm", "hmm",
"yall", "xoxo", "yolo", "em", "v", "ver", "hav", "vry", "shud", "wer",
"abt", "bc", "wen", "jus", "tht", "fr", "hs", "r", "wud", "cud"]

def PreProcess(tweet):
    #tweet = str(tweet)
    tweet = tweet.lower()
    tweet = BeautifulSoup(tweet, "lxml").get_text()
    tweet = re.sub("[^a-zA-Z]", " ", tweet)
    tweet = ' '.join([word for word in tweet.split() if word not in stops]) ## Remove eng stop words
    tweet = ' '.join([word for word in tweet.split() if word not in remove_words]) ## Remove specific words
    return tweet

import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
#from keras.engine.topology import Merge
from keras.layers import TimeDistributed, Lambda
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers.advanced_activations import PReLU
from keras.preprocessing import sequence, text
from keras.layers import Bidirectional
from keras.layers.normalization import BatchNormalization

data = pd.read_csv('train_E6oV3lV.csv')
#data.tweet = data.tweet.apply(PreProcess)

y = data.label.values

tk = text.Tokenizer(nb_words=200000)

max_len = 40


tk.fit_on_texts(data.tweet.values.astype(str))
x = tk.texts_to_sequences(data.tweet.values)
x = sequence.pad_sequences(x, maxlen=max_len)


word_index = tk.word_index
#ytrain_enc = np_utils.to_categorical(y)
embeddings_index = {}
f = open('glove.6B.300d.txt')
for line in tqdm(f):
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

max_features = 200000

embedding_layer = Embedding(len(word_index) + 1, 300, input_length=40, dropout=0.2)

model1 = Sequential()
model1.add(embedding_layer)
model1.add(((LSTM(100, dropout_W=0.2, dropout_U=0.2))))
model1.add(BatchNormalization())

model1.add(Dense(100))
model1.add(PReLU())
model1.add(Dropout(0.2))
model1.add(BatchNormalization())

model1.add(Dense(1))
model1.add(Activation('sigmoid'))


model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
checkpoint = ModelCheckpoint('weights.h5', monitor='val_acc', save_best_only=True, verbose=2)

model1.fit([x], y=y, batch_size=384, nb_epoch=10,
                 verbose=1, validation_split=0.1, shuffle=True, callbacks=[checkpoint])

