import gensim
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, BatchNormalization, Activation, Bidirectional
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.preprocessing.sequence import pad_sequences

import matplotlib as mpl
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
from keras.utils import plot_model 
from IPython.display import Image

# Specify the folder locations
W2V_DIR = '/Users/hareeshbahuleyan/Downloads/GoogleNews-vectors-negative300.bin'
DATA_DIR = '/Users/hareeshbahuleyan/Github/deeplearning-tutorials/text-classification-lstm/data/'

# These are some hyperparameters that can be tuned
MAX_SENT_LEN = 30
MAX_VOCAB_SIZE = 20000
LSTM_DIM = 128
EMBEDDING_DIM = 300
BATCH_SIZE = 32
N_EPOCHS = 10

# Read the text files of positive and negative sentences
with open(DATA_DIR+'rt-polarity.neg', 'r', errors='ignore') as f:
    neg = f.readlines()
    
with open(DATA_DIR+'rt-polarity.pos', 'r', errors='ignore') as f:
    pos = f.readlines()

print('Number of negative sentences:', len(neg))
print('Number of positive sentences:', len(pos))

# Create a dataframe to store the sentence and polarity as 2 columns
df = pd.DataFrame(columns=['sentence', 'polarity'])
df['sentence'] = neg + pos
df['polarity'] = [0]*len(neg) + [1]*len(pos)
df = df.sample(frac=1, random_state=10) # Shuffle the rows
df.reset_index(inplace=True, drop=True)

# Pre-processing involves removal of puctuations and converting text to lower case
word_seq = [text_to_word_sequence(sent) for sent in df['sentence']]
print('90th Percentile Sentence Length:', np.percentile([len(seq) for seq in word_seq], 90))

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts([' '.join(seq[:MAX_SENT_LEN]) for seq in word_seq])

print("Number of words in vocabulary:", len(tokenizer.word_index))

# Convert the sequence of words to sequnce of indices
X = tokenizer.texts_to_sequences([' '.join(seq[:MAX_SENT_LEN]) for seq in word_seq])
X = pad_sequences(X, maxlen=MAX_SENT_LEN, padding='post', truncating='post')

y = df['polarity']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size=0.1)

# Load the word2vec embeddings 
embeddings = gensim.models.KeyedVectors.load_word2vec_format(W2V_DIR, binary=True)

# Create an embedding matrix containing only the word's in our vocabulary
# If the word does not have a pre-trained embedding, then randomly initialize the embedding
embeddings_matrix = np.random.uniform(-0.05, 0.05, size=(len(tokenizer.word_index)+1, EMBEDDING_DIM)) # +1 is because the matrix indices start with 0
for word, i in tokenizer.word_index.items(): # i=0 is the embedding for the zero padding
    try:
        embeddings_vector = embeddings[word]
    except KeyError:
        embeddings_vector = None
    if embeddings_vector is not None:
        embeddings_matrix[i] = embeddings_vector
        
del embeddings

# Build a sequential model by stacking neural net units 
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index)+1,
                          output_dim=EMBEDDING_DIM,
                          weights = [embeddings_matrix], trainable=False, name='word_embedding_layer', 
                          mask_zero=True))

model.add(LSTM(LSTM_DIM, return_sequences=False, name='lstm_layer'))

model.add(Dense(1, activation='sigmoid', name='output_layer'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=N_EPOCHS,
          validation_data=(X_test, y_test))

score, acc = model.evaluate(X_test, y_test,
                            batch_size=BATCH_SIZE)

print("Accuracy on Test Set = {0:4.3f}".format(acc))

plot_model(model, to_file='basic_lstm_classifier.png', show_layer_names=True)
Image('basic_lstm_classifier.png')



# Build a sequential model by stacking neural net units 
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index)+1,
                          output_dim=EMBEDDING_DIM,
                          weights = [embeddings_matrix], trainable=False, name='word_embedding_layer', 
                          mask_zero=True)) # trainable=True results in overfitting

model.add(LSTM(LSTM_DIM, return_sequences=False, name='lstm_layer')) # Can try Bidirectional-LSTM

model.add(Dense(32, name='dense_1'))
# model.add(BatchNormalization(name='bn_1')) # BN did not really help with performance 
model.add(Dropout(rate=0.3, name='dropout_1')) # Can try varying dropout rates
model.add(Activation(activation='relu', name='activation_1'))

model.add(Dense(8, name='dense_2'))
# model.add(BatchNormalization(name='bn_2'))
model.add(Dropout(rate=0.3, name='dropout_2'))
model.add(Activation(activation='relu', name='activation_2'))

model.add(Dense(1, activation='sigmoid', name='output_layer'))
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=N_EPOCHS,
          validation_data=(X_test, y_test))

plot_model(model, to_file='modified_lstm_classifier.png', show_layer_names=True)
Image('modified_lstm_classifier.png')







