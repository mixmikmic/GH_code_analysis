import numpy as np
np.random.seed(420)

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

import matplotlib.pyplot as plt

plt.imshow(x_train[0],'gray')
plt.show()

print(y_train[0])

print(x_train.shape)
print(y_train.shape)

x_train = x_train.reshape((len(x_train),28,28,1))
print(x_train.shape)

from keras.utils import np_utils

y_train_onehot = np_utils.to_categorical(y_train,10)

print(y_train_onehot.shape)

x_test = x_test.reshape((len(x_test),28,28,1))
y_test_onehot = np_utils.to_categorical(y_test,10)

x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

x_train /= 255.
x_test /= 255.

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout, Activation

model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(28,28,1)))
model.add(Activation('relu'))
model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(128,activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))

print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train,y_train_onehot,epochs=10,validation_split=0.2,batch_size=128)

model.save('model.h5')

from keras.models import load_model

new_model = load_model('model.h5')

