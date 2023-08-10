from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import SGD

nb_classes = 10

# FC@512+relu -> DropOut(0.2) -> FC@512+relu -> DropOut(0.2) -> FC@nb_classes+softmax
# ... your Code Here

# Decomment and Execute this cell to get the solution
# %load solutions/sol_221_1.py

from keras.datasets import mnist
from keras.utils import np_utils

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

network_history = model.fit(X_train, Y_train, batch_size=128, 
                            epochs=100, verbose=1, validation_data=(X_test, Y_test))

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(network_history.history['loss'])
plt.plot(network_history.history['val_loss'])
plt.legend(['Training', 'Validation'])

plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(network_history.history['acc'])
plt.plot(network_history.history['val_acc'])
plt.legend(['Training', 'Validation'], loc='lower right')

# %load solutions/sol_221_2.py

