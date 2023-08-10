# load libraries
import numpy as np
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Set the number of features we want
number_of_features = 1000

# Load data and target vector from movie review data
(train_data, train_target), (test_data, test_target) = imdb.load_data(num_words=number_of_features)

# Convert movie review data to a one-hot encoded feature matrix
tokenizer = Tokenizer(num_words=number_of_features)
train_features = tokenizer.sequences_to_matrix(train_data, mode='binary')
test_features = tokenizer.sequences_to_matrix(test_data, mode='binary')

# Start neural netwok
network = models.Sequential()

# Add fully connected layer with a ReLU activation function
network.add(layers.Dense(units=16, activation='relu', input_shape=(number_of_features,)))

# Add fully connected layer with a ReLU activation function
network.add(layers.Dense(units=16, activation='relu'))

# Add fully connected layer with a sigmoid activation function
network.add(layers.Dense(units=1, activation='sigmoid'))

# Compile neural network
network.compile(loss='binary_crossentropy', # cross-entropy
               optimizer = 'rmsprop', # root mean square propagation
               metrics = ['accuracy']) # accuracy performance metric

# Set callback function to early stop training and save the best so far
callbacks = [EarlyStopping(monitor='val_loss', patience=2),
            ModelCheckpoint(filepath='best_model.h5', monitor='val_loss',
                           save_best_only=True)]

# Train neural network
history = network.fit(train_features, # features
                     train_target, # target vector
                     epochs =20, # number of epochs
                     callbacks = callbacks, # early stopping
                     verbose = 1, # print description after each epochs
                     batch_size = 100, # number of observation per batch
                     validation_data = (test_features, test_target)) # data for evaluation



