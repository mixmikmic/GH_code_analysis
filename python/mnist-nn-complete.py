# Import MNIST dataset from Keras

from keras.datasets import mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()

# Data exploration

print("Inputs shape is " + str(train_x.shape)) # 60,000 samples, each image: 28 x 28 pixels
print("Input type is " + str(type(train_x)))
print("Labels:")
print(train_y)
print("Labels shape is " + str(train_y.shape))
print("Labels type is " + str(type(train_y)))

# Matplotlib: Data visualization library
import matplotlib.pyplot as plt

# Visualize the input samples

sample_num = 0 # change this number and re-run the cell to see different image samples!

plt.imshow(train_x[sample_num], cmap=plt.get_cmap('gray'))
print(train_y[sample_num])
plt.show()

# Flatten 28*28 images to a 784 vector for each image

num_pixels = train_x.shape[1] * train_x.shape[2] # 28 * 28 = 784
train_x_flattened = train_x.reshape(train_x.shape[0], num_pixels).astype('float32') # new shape: 60,000 x 784
test_x_flattened = test_x.reshape(test_x.shape[0], num_pixels).astype('float32') # new shape: 10,000 x 784

# Normalize pixel values to between 0-1
train_x_flattened = train_x_flattened / 255.
test_x_flattened = test_x_flattened / 255.

import keras

# Use Keras to categorize the outputs ("one-hot" vectors)
train_y_categorical = keras.utils.to_categorical(train_y, num_classes=10)
test_y_categorical = keras.utils.to_categorical(test_y, num_classes=10)

# let's see result of categorizing the outputs
print(train_y_categorical[:5]) # print out first 5 training label vectors

from keras.layers import Dense, Activation
from keras.models import Sequential

# Initialize simple neural network model
model = Sequential()

# NOTE: This model contains some tweaks from the original 'incomplete' notebook in order to improve performance.
    # 1. Use 'relu' activation instead of 'sigmoid' for hidden layer
    # 2. Use 'softmax' activation for final output layer
    # 3. Use 'categorical crossentropy' loss function, instead of 'mean-squared-error'
    # See the links provided below for more info

# Hidden layer 1: 64 neurons, 'relu' activation 
    # (see: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
    # http://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html
model.add(Dense(units=64, input_dim=784, activation='relu'))

# Hidden layer 2: 32 neurons, 'relu' activation
model.add(Dense(units=32, activation='relu'))

# Output layer: 10 neurons (one for each class), 'softmax' activation
    # This layer represents the scores that the network assigns to each possible digit, 1-10
    # See: http://dataaspirant.com/2017/03/07/difference-between-softmax-function-and-sigmoid-function/
model.add(Dense(units=10, activation='softmax'))

# Compile the model, get ready to train

    # Loss: Categorical Crossentropy
        # See: http://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
    # Optimizer: stochastic gradient descent (SGD)
    # Additional metrics: Accuracy

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    

# Print model summary
model.summary()

# Fit the model to the training data

# Train the model
    # Number of epochs: 10 (i.e. how many times to loop over the training data)
    # Batch size: 16 (how big our "drunk walk" samples should be)
    # See: 'fit()' in https://keras.io/models/sequential/
    
model.fit(train_x_flattened, train_y_categorical, epochs=10, batch_size=16)
    

# Evaluate trained model on test data

# Returns final test loss & test accuracy
    # See: 'evaluate' in https://keras.io/models/sequential/
loss_and_metrics = model.evaluate(test_x_flattened, test_y_categorical, batch_size=128)
final_cost = loss_and_metrics[0]
final_accuracy = loss_and_metrics[1]

print()
print("Final test cost: ", final_cost)
print("Final test accuracy: ", final_accuracy)

import numpy as np

sample_num = 10 # which test sample to look at. TODO: Play around with this number to see how \
    # our neural network performs on different test images

# Predicted class
test_sample = np.expand_dims(test_x_flattened[sample_num], axis=0) # pick out a one-sample "batch" to feed into model
predicted_scores = model.predict(test_sample) # outputted probabilities vector
print("Output vector: ", predicted_scores[0]) # print predicted scores

predicted_class = np.argmax(predicted_scores) # pick the class with highest probability --> final prediction
print("Predicted digit: ", predicted_class) # print predicted classification

# Show actual input image
plt.imshow(test_x[sample_num], cmap=plt.get_cmap('gray'))
plt.show()



