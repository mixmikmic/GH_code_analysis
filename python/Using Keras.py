import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from matplotlib import pyplot as plt
import theano
import time

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Visulization of data
plt.imshow(X_train[0], cmap = 'gray')

X_train = X_train.reshape(X_train.shape[0],28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_test.shape

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_test /= 255
X_train /= 255

print(y_train.shape)
print(y_test.shape[0])

# Convert one dimentional class arrays to 10-dimentional class matrix
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

y_test

print(y_test.shape)

model = Sequential()

# CNN input layer
model.add(Convolution2D(32, 3, 3, activation = 'relu', input_shape = (28, 28, 1)))

print(model.output_shape)

model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#Fully connected dense layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(X_train, y_train, batch_size=32, nb_epoch = 10, verbose = 1)

score = model.evaluate(X_test, y_test, verbose=0)

print(score)

# trial = model.predict(X_test[0])
trial = model.predict(X_test)

s = 0
for i in range(len(X_test)):
    image_num = i
#     plt.imshow(X_test[image_num].reshape(28,28), cmap = 'gray')
#     plt.title('Image')
    pred = np.argmax(trial[image_num])
    act = np.argmax(y_test[image_num])
#     print('The predicted number by CNN is', pred)
#     print('Actual label of the image is', act)
    if pred == act:
        print('The guess was correct')
    else:
        print('The guess was wrong')
        plt.imshow(X_test[image_num].reshape(28,28), cmap = 'gray')
        plt.title('Image')
        print('The predicted number by CNN is', pred)
        print('Actual label of the image is', act)
        s += 1

print('Number of incorrect guesses = ',s, 'out of', len(X_test))

