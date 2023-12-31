from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.utils import np_utils

batch_size = 128
nb_classes = 10
nb_epoch = 10

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
Y_Train = np_utils.to_categorical(y_train, nb_classes)
Y_Test = np_utils.to_categorical(y_test, nb_classes)

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import random

samples = np.concatenate([np.concatenate([X_train[i].reshape(28,28) for i in [int(random.random() * len(X_train)) for i in range(16)]], axis=1) for i in range(2)], axis=0)
plt.figure(figsize=(16,2))
plt.imshow(samples, cmap='gray')

# Multilayer Perceptron model
model = Sequential()
model.add( Dense(input_dim=784, units=625, kernel_initializer="normal", activation="sigmoid") )
model.add( Dense(input_dim=625, units=625, kernel_initializer="normal", activation="sigmoid") )
model.add( Dense(input_dim=625, units=10,  kernel_initializer="normal", activation="softmax") )
model.compile( optimizer=SGD(lr=0.05), loss='categorical_crossentropy', metrics=['accuracy'] )
model.summary()

# Train
history = model.fit( X_train, Y_Train, epochs=nb_epoch, batch_size=batch_size, verbose=1 )

# Evaluate
evaluation = model.evaluate(X_test, Y_Test, verbose=1)
print('Summary: Loss over the test dataset: %.2f, Accuracy: %.2f' % (evaluation[0], evaluation[1]))

