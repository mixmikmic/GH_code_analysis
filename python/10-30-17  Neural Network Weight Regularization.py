# load libraries
import numpy as np
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
from keras import regularizers

# Set random seed
np.random.seed(0)

# Set the number of features we want
number_of_features = 1000

# Load data and target vector from movie review data
(train_data, train_target), (test_data, test_target) = imdb.load_data(num_words=number_of_features)

# Convert movie review data to a one-hot encoded feature matrix
tokenizer = Tokenizer(num_words=number_of_features)
train_features = tokenizer.sequences_to_matrix(train_data, mode='binary')
test_features = tokenizer.sequences_to_matrix(test_data, mode='binary')

# Start neural network
network = models.Sequential()

# Add fully connected layer with a ReLU activation functions and L2 regularization
network.add(layers.Dense(units=16, activation='relu', 
                         kernel_regularizer=regularizers.l2(0.01),
                        input_shape = (number_of_features,)))

# Add fully connected layer with a ReLU activation function and L2 regularization
network.add(layers.Dense(units=16, kernel_regularizer=regularizers.l2(0.01),
                       input_shape=(number_of_features,)))

# Add fully connected layer with a sigmoid activation function
network.add(layers.Dense(units=1, activation='sigmoid'))

# Compile neural network
network.compile(loss='binary_crossentropy', # Cross-entropy
               optimizer='rmsprop', # cross-entropy
                metrics=['accuracy'])

# Train neural network
history = network.fit(train_features, # features
                     train_target, # target vector
                     epochs = 3, # number of epochs
                     verbose = 1, # Show output
                     batch_size=100, # number of observations per batch
                     validation_data = (test_features, test_target)) # data fro evaluation

