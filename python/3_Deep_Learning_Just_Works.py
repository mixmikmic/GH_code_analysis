import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, MaxPool2D, Conv2D, Flatten
from keras.optimizers import Adam
get_ipython().run_line_magic('matplotlib', 'inline')


# Prepare the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((-1, 28, 28, 1)) / 255.0
x_test = x_test.reshape((-1, 28, 28, 1)) / 255.0
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)


model = Sequential()
model.add(Conv2D(64, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(128, (5, 5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])

results = model.fit(x_train, y_train,
          batch_size=256,
          epochs=30,
          verbose=2,
          validation_data=(x_test, y_test))

plt.figure(1)
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.legend(['train loss', 'test loss'])

plt.figure(2)
plt.plot(results.history['acc'])
plt.plot(results.history['val_acc'])
plt.legend(['train acc', ' acc'])

print("Train error (%): ", 100  - model.evaluate(x_train, y_train, batch_size=128)[1]*100)
print("Test error (%): ", 100- model.evaluate(x_test, y_test, batch_size=128)[1]*100)

model.summary()

import keras.backend as K

get_conv_outputs = K.function([model.layers[0].input, K.learning_phase()], [model.layers[0].output, model.layers[2].output])

[conv1, conv2] = get_conv_outputs([x_train[0:2], 0])

plt.figure()
plt.imshow(x_train[0].reshape(28,28), cmap='gray')
plt.title('Input image: ')

plt.figure(figsize=(24, 24))
for i in range(64):
    plt.subplot(8, 8, i+1)
    plt.imshow(conv1[0, :, :, i], cmap='gray')
    
plt.figure(figsize=(24, 24))   
for i in range(128):
    plt.subplot(16, 8, i+1)
    plt.imshow(conv2[0, :, :, i], cmap='gray')    

plt.figure()
plt.imshow(x_train[1].reshape(28,28), cmap='gray')
plt.title('Input image: ')

plt.figure(figsize=(24, 24))
for i in range(64):
    plt.subplot(8, 8, i+1)
    plt.imshow(conv1[1, :, :, i], cmap='gray')
    
plt.figure(figsize=(24, 24))   
for i in range(128):
    plt.subplot(16, 8, i+1)
    plt.imshow(conv2[1, :, :, i], cmap='gray')    

