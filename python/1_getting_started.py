import keras
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Train/test data shape:", x_train.shape, x_test.shape)
print("Train/test labels shape:", y_train.shape, y_test.shape)

import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

for i in range(3):
    plt.figure(i)
    plt.imshow(x_train[i], cmap='gray')
    plt.title("Image: " + str(i) + " Label: " + str(y_train[i]))


# Reshape and normalize the data
x_train = x_train.reshape((-1, 28*28)) / 255.0
x_test = x_test.reshape((-1, 28*28)) / 255.0

# Encode the labels
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

print("Train/test data shape:", x_train.shape, x_test.shape)
print("Train/test labels shape:", y_train.shape, y_test.shape)
print(y_train[0], np.argmax(y_train[0]))
print(y_train[0], np.argmax(y_train[1]))
print(y_train[0], np.argmax(y_train[2]))

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam

model = Sequential()

# For the first layer we have to define the input dimensionality
model.add(Dense(64, activation='relu', input_dim=784))
# Add a second hidden layer
model.add(Dense(256, activation='relu'))
# Add an output layer (the number of neurons must match the number of classes)
model.add(Dense(10, activation='softmax'))

# Select an optimizer
adam = Adam(lr=0.0001)
# Select the loss function and metrics that should be monitored
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=50, batch_size=256, verbose=2)

print("Train accuracy: ", model.evaluate(x_train, y_train, batch_size=128))
print("Test accuracy: ", model.evaluate(x_test, y_test, batch_size=128))

y_out = model.predict(x_train)
print(y_out[0], np.argmax(y_out[0]))

for i in range(10):
    print("Prediction order: ", np.argsort(y_out[i])[::-1], "True label: ", np.argmax(y_train[i]))

