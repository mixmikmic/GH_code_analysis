import os
# disable tensorflow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Set backend of Keras to Tensorflow not Theano
os.environ['KERAS_BACKEND'] = 'tensorflow'

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, pooling
from keras.utils import np_utils
# MNIST DATA Module
from keras.datasets import mnist

# Load MNIST Data
# if you don't have MNIST, it would be downloaded
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Shape of MNIST input data of Training Set
print(X_train.shape)

# Reshaping of input data for training with MLP(Multi-Layer Perceptron)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[-2] * X_train.shape[-1]) # 60000, 28 * 28
X_test = X_test.reshape(X_test.shape[0], X_test.shape[-2] * X_test.shape[-1]) # 10000, 28 * 28
print(X_train.shape)
print(X_test.shape)

# Modifying type of inputs for train; uint8 -> float32
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalization with inputs. Max value 255; Min value 0
X_train /= 255.0
X_test /= 255.0

# One hot encoding with labels
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

# Model Initialization
model = Sequential()

## Dense Layer 784 -> 128 node, Activation Function = ReLU
# activation: string, [*relu, sigmoid, tanh]
model.add(Dense(128, input_dim=784, activation='relu'))

# Dropout
model.add(Dropout(0.5))

# Dense Layer 128 -> 10 node, Activation Function = Softmax => Output
model.add(Dense(10, activation='softmax'))

## Loss Function = Cross Entropy, Optimizer = Adam Optimizer
# optimizer: [adam, rmsprop, adagrad, adadelta, sgd, adamax, *nadam]
# Default Learning Rate: adam=0.001, rmsprop=0.001, adagrad=0.01, adadelta=1.0, sgd=0.01, adamax=0.002, nadam=0.002
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

## Model Train
# Input: X_train, Label: Y_train
# Batch size = 32 ; don't increase this too large. It affects GPU Memory Usage
# verbose: 0 for no logging to stdout, 1 for progress bar logging, 2 for one log line per epoch.
model.fit(X_train, Y_train, batch_size=32, epochs=1, verbose=1)

# Test Set Accuracy Evaluation
score = model.evaluate(X_test, Y_test, verbose=0)
print(model.metrics_names)
print(score)

# Training Accuracy Evaluation
score = model.evaluate(X_train, Y_train, verbose=0)
print(model.metrics_names)
print(score)



