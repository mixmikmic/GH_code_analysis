from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.optimizers import SGD,Adam
import numpy as np
import matplotlib.pyplot as plt


batch_size = 128
num_classes = 10 # 10 classes because we have 10 digits (0-9)
epochs = 5
learning_rate = 0.01

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape 2d (28x28) image data into 1d vectors (28x28 = 784-d vectors)
x_train = x_train.reshape(x_train.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)

# set type to float, then clamp data to values between 0-1 instead of integers 0 - 255
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class labels into vectors. Ie 3 -> [0,0,0,1,0,0,0,0,0,0]
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# make model here

learning_rate = 0.1
# compile model here

# fit model here
# evaluate model 
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# write and compile the fancy model 

# fit the fancy model 
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

def predict(n):
    """Predict the nth test data sample using the model"""
    print("Prediction: " + str(np.argmax(np.round(model.predict(np.expand_dims(x_test[n], 0)), 2))))
    print("Actual: " + str(np.argmax(y_test[n])))
    plt.imshow(x_test[n].reshape(28,28), cmap='Greys')
    plt.show()
    
predict(2354)

batch_size = 128
num_classes = 10 # 10 classes because we have 10 digits (0-9)
epochs = 5

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# set type to float, then clamp data to values between 0-1 instead of integers 0 - 255
x_train = x_train.reshape(list(x_train.shape) + [1])
x_test = x_test.reshape(list(x_test.shape) + [1])
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# convert class labels into vectors. Ie 3 -> [0,0,0,1,0,0,0,0,0,0]
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = None

# define and compile CNN model

# Fit the CNN model 
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



