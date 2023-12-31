import numpy as np

from sklearn.metrics import accuracy_score
from keras.datasets import cifar10
from keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, Input, Activation, Reshape, concatenate
from keras import optimizers

model = Sequential()

model.add(Conv2D(input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

print(model.output_shape)

model.add(Reshape(target_shape = (16*16, 50)))

model.add(LSTM(50, return_sequences = False))

model.add(Dense(10))
model.add(Activation('softmax'))

adam = optimizers.Adam(lr = 0.001)
model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])

print(model.summary())

get_ipython().run_cell_magic('time', '', 'history = model.fit(X_train, y_train, epochs = 100, batch_size = 100, verbose = 0)')

results = model.evaluate(X_test, y_test)

print('Test Accuracy: ', results[1])

input_layer = Input(shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]))
conv_layer = Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same')(input_layer)
activation_layer = Activation('relu')(conv_layer)
pooling_layer = MaxPooling2D(pool_size = (2,2), padding = 'same')(activation_layer)
flatten = Flatten()(pooling_layer)
dense_layer_1 = Dense(100)(flatten)

reshape = Reshape(target_shape = (X_train.shape[1]*X_train.shape[2], X_train.shape[3]))(input_layer)
lstm_layer = LSTM(50, return_sequences = False)(reshape)
dense_layer_2 = Dense(100)(lstm_layer)

merged_layer = concatenate([dense_layer_1, dense_layer_2])

output_layer = Dense(10, activation = 'softmax')(merged_layer)

model = Model(inputs = input_layer, outputs = output_layer)

adam = optimizers.Adam(lr = 0.001)
model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])

print(model.summary())

get_ipython().run_cell_magic('time', '', 'history = model.fit(X_train, y_train, epochs = 10, batch_size = 100, verbose = 0)')

results = model.evaluate(X_test, y_test)

print('Test Accuracy: ', results[1])

