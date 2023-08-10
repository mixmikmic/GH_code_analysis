import sys; print('Python \t\t{0[0]}.{0[1]}'.format(sys.version_info))
import tensorflow as tf; print('Tensorflow \t{}'.format(tf.__version__))
import keras; print('Keras \t\t{}'.format(keras.__version__))

get_ipython().magic('matplotlib inline')

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../mnist-data/", one_hot=True)

mnist.train.images.shape

plt.figure(figsize=(15,5))
for i in list(range(10)):
    plt.subplot(1, 10, i+1)
    pixels = mnist.test.images[i]
    pixels = pixels.reshape((28, 28))
    plt.imshow(pixels, cmap='gray_r')
plt.show()

from keras.models import Model
from keras.layers import Input, Dense, Activation
from keras.layers import Dropout, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import BatchNormalization

def mlp(batch_normalization=False, activation='sigmoid'):
    _in = Input(shape=(784,))
    
    for i in range(5):
        x = Dense(128, activation=activation, input_shape=(784,))(x if i else _in)
        if batch_normalization:
            x = BatchNormalization()(x)

    _out = Dense(10, activation='softmax')(x)
    model = Model(_in, _out)

    return model

from  functools import reduce

def print_layers(model):
    for l in model.layers:
        print(l.name, l.output_shape, [reduce(lambda x, y: x*y, w.shape) for w in l.get_weights()])

from keras.callbacks import Callback

class BatchLogger(Callback):
    def on_train_begin(self, epoch, logs={}):
        self.log_values = {}
        for k in self.params['metrics']:
            self.log_values[k] = []

    def on_batch_end(self, batch, logs={}):
        for k in self.params['metrics']:
            if k in logs:
                self.log_values[k].append(logs[k])

# see http://cs231n.github.io/neural-networks-3/

model = mlp(False, 'sigmoid')
print_layers(model)

bl_noBN = BatchLogger()

from keras.optimizers import Adam
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=["accuracy"])

model.fit(mnist.train.images, mnist.train.labels,
          batch_size=128, epochs=1, verbose=1, callbacks=[bl_noBN],
          validation_data=(mnist.test.images, mnist.test.labels))

# see http://cs231n.github.io/neural-networks-3/

model = mlp(True, 'sigmoid')
print_layers(model)

bl_BN = BatchLogger()

from keras.optimizers import Adam
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=["accuracy"])

model.fit(mnist.train.images, mnist.train.labels,
          batch_size=128, epochs=1, verbose=1, callbacks=[bl_BN],
          validation_data=(mnist.test.images, mnist.test.labels))

plt.figure(figsize=(15,5))
plt.subplot(1, 2, 1)
plt.title('loss, per batch')
plt.plot(bl_noBN.log_values['loss'], 'b-', label='no BN');
plt.plot(bl_BN.log_values['loss'], 'r-', label='with BN');
plt.legend(loc='upper right')
plt.subplot(1, 2, 2)
plt.title('accuracy, per batch')
plt.plot(bl_noBN.log_values['acc'], 'b-', label='no BN');
plt.plot(bl_BN.log_values['acc'], 'r-', label='with BN');
plt.legend(loc='lower right')
plt.show()

# see http://cs231n.github.io/neural-networks-3/

model = mlp(False, 'relu')
print_layers(model)

bl_noBN = BatchLogger()

from keras.optimizers import Adam
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=["accuracy"])

model.fit(mnist.train.images, mnist.train.labels,
          batch_size=128, epochs=1, verbose=1, callbacks=[bl_noBN],
          validation_data=(mnist.test.images, mnist.test.labels))

# see http://cs231n.github.io/neural-networks-3/

model = mlp(True, 'relu')
print_layers(model)

bl_BN = BatchLogger()

from keras.optimizers import Adam
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=["accuracy"])

model.fit(mnist.train.images, mnist.train.labels,
          batch_size=128, epochs=1, verbose=1, callbacks=[bl_BN],
          validation_data=(mnist.test.images, mnist.test.labels))

plt.figure(figsize=(15,5))
plt.subplot(1, 2, 1)
plt.title('loss, per batch')
plt.plot(bl_noBN.log_values['loss'], 'b-', label='no BN');
plt.plot(bl_BN.log_values['loss'], 'r-', label='with BN');
plt.legend(loc='upper right')
plt.subplot(1, 2, 2)
plt.title('accuracy, per batch')
plt.plot(bl_noBN.log_values['acc'], 'b-', label='no BN');
plt.plot(bl_BN.log_values['acc'], 'r-', label='with BN');
plt.legend(loc='lower right')
plt.show()



