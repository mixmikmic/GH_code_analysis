import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
get_ipython().run_line_magic('matplotlib', 'inline')


# Prepare the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((-1, 28*28)) / 255.0
x_test = x_test.reshape((-1, 28*28)) / 255.0
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)


model = Sequential()

model.add(Dense(256, activation='sigmoid', input_dim=784, kernel_initializer='random_uniform'))
model.add(Dense(256, activation='sigmoid', kernel_initializer='random_uniform'))
model.add(Dense(512, activation='sigmoid', kernel_initializer='random_uniform'))
model.add(Dense(512, activation='sigmoid', kernel_initializer='random_uniform'))
model.add(Dense(1024, activation='sigmoid', kernel_initializer='random_uniform'))
model.add(Dense(1024, activation='sigmoid', kernel_initializer='random_uniform'))

model.add(Dense(10, activation='softmax'))
from keras.optimizers import SGD
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.001, momentum=0.9), metrics=['accuracy'])

results = model.fit(x_train, y_train, epochs=50, batch_size=256, verbose=0, validation_data=(x_test, y_test))
print("Train accuracy: ", model.evaluate(x_train, y_train, batch_size=128))
print("Test accuracy: ", model.evaluate(x_test, y_test, batch_size=128))

plt.figure(1)
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.legend(['train loss', 'test loss'])

model = Sequential()

model.add(Dense(256, activation='sigmoid', input_dim=784, kernel_initializer='random_uniform'))
model.add(Dense(10, activation='softmax'))
from keras.optimizers import SGD
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.001, momentum=0.9), metrics=['accuracy'])

results = model.fit(x_train, y_train, epochs=50, batch_size=256, verbose=2, validation_data=(x_test, y_test))
print("Train accuracy: ", model.evaluate(x_train, y_train, batch_size=128))
print("Test accuracy: ", model.evaluate(x_test, y_test, batch_size=128))

plt.figure(1)
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.legend(['train loss', 'test loss'])

model = Sequential()

model.add(Dense(256, activation='relu', input_dim=784, kernel_initializer=keras.initializers.he_uniform()))
model.add(Dense(256, activation='relu', kernel_initializer=keras.initializers.he_uniform()))
model.add(Dense(512, activation='relu', kernel_initializer=keras.initializers.he_uniform()))
model.add(Dense(512, activation='relu', kernel_initializer=keras.initializers.he_uniform()))
model.add(Dense(1024, activation='relu', kernel_initializer=keras.initializers.he_uniform()))
model.add(Dense(1024, activation='relu', kernel_initializer=keras.initializers.he_uniform()))

model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])

model.summary()

results = model.fit(x_train, y_train, epochs=50, batch_size=256, verbose=2, validation_data=(x_test, y_test))

print("Log data: ", results.history.keys())

plt.figure(1)
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.legend(['train loss', 'test loss'])

plt.figure(2)
plt.plot(results.history['acc'])
plt.plot(results.history['val_acc'])
plt.legend(['train acc', ' acc'])

