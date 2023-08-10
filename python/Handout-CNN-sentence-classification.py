import numpy as np
import pylab as pl
from IPython.display import SVG
from os.path import join, exists, split
import os

from gensim.models import word2vec, KeyedVectors


from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding, GlobalMaxPooling1D
from keras.layers.merge import Concatenate
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.utils.vis_utils import model_to_dot

get_ipython().run_line_magic('matplotlib', 'inline')

# Model Hyperparameters
embedding_dim = 300
filter_sizes = (3, 4, 5 )
num_filters = 100
dropout_prob = (0.0, 0.5)

# Training parameters
batch_size = 64
num_epochs = 10

# Prepossessing parameters
sequence_length = 20
max_words = 5000

# Word2Vec parameters 
min_word_count = 1
context = 10

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words, start_char=None,
                                                      oov_char=None, index_from=None)
x_train = x_train[:1500]
y_train = y_train[:1500]
x_test = x_test[:10000]
y_test = y_test[:10000]

x_train = sequence.pad_sequences(x_train, maxlen=sequence_length, padding="post", truncating="post")
x_test = sequence.pad_sequences(x_test, maxlen=sequence_length, padding="post", truncating="post")

vocabulary = imdb.get_word_index()
vocabulary_inv = dict((v, k) for k, v in vocabulary.items())
vocabulary_inv[0] = "<PAD/>"

print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("Vocabulary Size: {:d}".format(len(vocabulary_inv)))

embedding_model = KeyedVectors.load_word2vec_format('GNvectors.bin', binary=True)
embedding_weights = {key: embedding_model[word] if word in embedding_model else
                          np.random.uniform(-0.25, 0.25, embedding_model.vector_size)
                     for key, word in vocabulary_inv.items()}

# Input layer
input_shape = (sequence_length,)
model_input = Input(shape=input_shape)

# Embedding layer
embedding_layer = Embedding(len(vocabulary_inv), embedding_dim, input_length=sequence_length, name="embedding")
z = embedding_layer(model_input)

# Convolutional block
conv_blocks = []
for sz in filter_sizes:
    conv = Convolution1D(filters=num_filters,
                         kernel_size=sz,
                         padding="valid",
                         activation="relu",
                         strides=1)(z)
    conv = GlobalMaxPooling1D()(conv)
    conv_blocks.append(conv)
z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

# Dropout 
z = Dropout(dropout_prob[1])(z)

# Output layer
model_output = Dense(1, activation="sigmoid")(z)

# Model compilation
model_rand = Model(model_input, model_output)
model_rand.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model_rand.summary(85)
SVG(model_to_dot(model_rand, show_shapes=True).create(prog='dot', format='svg'))

num_epochs = 10
history_rand = model_rand.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs,
          validation_data=(x_test, y_test), verbose=2)

pl.plot(history_rand.history['loss'], label='loss')
pl.plot(history_rand.history['val_loss'], label='val_loss')
pl.legend()
pl.xlabel('Epoch')
pl.ylabel('Loss')

pl.plot(history_rand.history['acc'], label='acc')
pl.plot(history_rand.history['val_acc'], label='val_acc')
pl.legend()
pl.xlabel('Epoch')
pl.ylabel('Accuracy')

# Input layer
input_shape = (sequence_length,)
model_input = Input(shape=input_shape)

# Embedding layer
embedding_layer = Embedding(len(vocabulary_inv), embedding_dim, input_length=sequence_length, name="embedding")
z = embedding_layer(model_input)
weights = np.array([embedding_weights[i] for i in range(len(embedding_weights))])
print("Initializing embedding layer with word2vec weights, shape", weights.shape)
embedding_layer.set_weights([weights])

# Weights is embedding layer are not modified during training
embedding_layer.trainable = False

# Convolutional block
conv_blocks = []
for sz in filter_sizes:
    conv = Convolution1D(filters=num_filters,
                         kernel_size=sz,
                         padding="valid",
                         activation="relu",
                         strides=1)(z)
    conv = GlobalMaxPooling1D()(conv)
    conv_blocks.append(conv)
z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

# Dropout 
z = Dropout(dropout_prob[1])(z)

# Output layer
model_output = Dense(1, activation="sigmoid")(z)

# Model compilation
model_static = Model(model_input, model_output)
model_static.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model_static.summary(85)

num_epochs = 10
history_static = model_static.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs,
          validation_data=(x_test, y_test), verbose=2)

pl.plot(history_static.history['loss'], label='loss')
pl.plot(history_static.history['val_loss'], label='val_loss')
pl.legend()
pl.xlabel('Epoch')
pl.ylabel('Loss')

pl.plot(history_static.history['acc'], label='acc')
pl.plot(history_static.history['val_acc'], label='val_acc')
pl.legend()
pl.xlabel('Epoch')
pl.ylabel('Accuracy')

# Input layer
input_shape = (sequence_length,)
model_input = Input(shape=input_shape)

# Embedding layer
embedding_layer = Embedding(len(vocabulary_inv), embedding_dim, input_length=sequence_length, name="embedding")
z = embedding_layer(model_input)
weights = np.array([embedding_weights[i] for i in range(len(embedding_weights))])
print("Initializing embedding layer with word2vec weights, shape", weights.shape)
embedding_layer.set_weights([weights])

# Weights is embedding layer are not modified during training
embedding_layer.trainable = True

# Convolutional block
conv_blocks = []
for sz in filter_sizes:
    conv = Convolution1D(filters=num_filters,
                         kernel_size=sz,
                         padding="valid",
                         activation="relu",
                         strides=1)(z)
    conv = GlobalMaxPooling1D()(conv)
    conv_blocks.append(conv)
z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

# Dropout 
z = Dropout(dropout_prob[1])(z)

# Output layer
model_output = Dense(1, activation="sigmoid")(z)

# Model compilation
model_non_static = Model(model_input, model_output)
model_non_static.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

num_epochs = 10
history_non_static = model_non_static.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs,
          validation_data=(x_test, y_test), verbose=2)

pl.plot(history_non_static.history['loss'], label='loss')
pl.plot(history_non_static.history['val_loss'], label='val_loss')
pl.legend()
pl.xlabel('Epoch')
pl.ylabel('Loss')

pl.plot(history_non_static.history['acc'], label='acc')
pl.plot(history_non_static.history['val_acc'], label='val_acc')
pl.legend()
pl.xlabel('Epoch')
pl.ylabel('Accuracy')

pl.plot(history_rand.history['val_acc'], label='rand')
pl.plot(history_static.history['val_acc'], label='static')
pl.plot(history_non_static.history['val_acc'], label='non_static')
pl.legend()
pl.xlabel('Epoch')
pl.ylabel('Validation Accuracy')

from keras import backend as K

def re_initialize(model):
    session = K.get_session()
    for layer in model.layers: 
         for v in layer.__dict__:
             v_arg = getattr(layer,v)
             if hasattr(v_arg,'initializer'):
                 initializer_method = getattr(v_arg, 'initializer')
                 initializer_method.run(session=session)
                 print('reinitializing layer {}.{}'.format(layer.name, v))

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words, start_char=None,
                                                      oov_char=None, index_from=None)

x_test = x_test[:10000]
y_test = y_test[:10000]

x_train = sequence.pad_sequences(x_train, maxlen=sequence_length, padding="post", truncating="post")
x_test = sequence.pad_sequences(x_test, maxlen=sequence_length, padding="post", truncating="post")

vocabulary = imdb.get_word_index()
vocabulary_inv = dict((v, k) for k, v in vocabulary.items())
vocabulary_inv[0] = "<PAD/>"

print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("Vocabulary Size: {:d}".format(len(vocabulary_inv)))

num_epochs = 10
re_initialize(model_rand)
history_rand = model_rand.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs,
          validation_data=(x_test, y_test), verbose=2)

num_epochs = 10
re_initialize(model_static)
embedding_layer = model_static.get_layer("embedding")
embedding_layer.set_weights([weights])
history_static = model_static.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs,
          validation_data=(x_test, y_test), verbose=2)

num_epochs = 10
re_initialize(model_non_static)
embedding_layer = model_non_static.get_layer("embedding")
embedding_layer.set_weights([weights])
history_non_static = model_non_static.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs,
          validation_data=(x_test, y_test), verbose=2)

pl.plot(history_rand.history['val_acc'], label='rand')
pl.plot(history_static.history['val_acc'], label='static')
pl.plot(history_non_static.history['val_acc'], label='non_static')
pl.legend()
pl.xlabel('Epoch')
pl.ylabel('Validation Accuracy')



