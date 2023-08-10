# Load libraries
import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import models
from keras import layers

# Set random seed
np.random.seed(0)

# Set the number of features we want
number_of_features = 1000

# Load data and target vector from movie review data
(train_data, train_target), (test_data, test_target) = imdb.load_data(num_words=number_of_features)

# Use padding or truncate to make each observation have 400 features
train_features = sequence.pad_sequences(train_data, maxlen=400)
test_features = sequence.pad_sequences(test_data, maxlen=400)

# Viwe first observation
print(train_data[0])

# View first observation
test_features[0]

# Start a neural network
network = models.Sequential()

# Add an embedding layer
network.add(layers.Embedding(input_dim = number_of_features, output_dim=128))

# Add a long short-term memory layer with 128 units
network.add(layers.LSTM(units=200))

# Add fully connected layer with a sigmoid activation function
network.add(layers.Dense(units=1, activation='sigmoid'))

# Compile neural network
network.compile(loss='binary_crossentropy', # cross-entropy
               optimizer = 'Adam', # Adam optimization
               metrics=['accuracy']) # Accuracy Performance metric

# Train neural network
history = network.fit(train_features, # Features
                     train_target, # Target
                     epochs = 3, # Number of Epochs
                     verbose = 1, # show description after each epoch
                     batch_size = 1000, # number of observation per batch
                     validation_data = (test_features, test_target)) # Data for evaluation

