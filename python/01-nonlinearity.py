from __future__ import division, print_function
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.utils import np_utils

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

def read_dataset(filename):
    Z = np.loadtxt(filename, delimiter=",")
    y = Z[:, 0]
    X = Z[:, 1:]
    return X, y

def plot_dataset(X, y):
    Xred = X[y==0]
    Xblue = X[y==1]
    plt.scatter(Xred[:, 0], Xred[:, 1], color='r', marker='o')
    plt.scatter(Xblue[:, 0], Xblue[:, 1], color='b', marker='o')
    plt.xlabel("X[0]")
    plt.ylabel("X[1]")
    plt.show()

X, y = read_dataset("../data/linear.csv")
X = X[y != 2]
y = y[y != 2].astype("int")
print(X.shape, y.shape)
plot_dataset(X, y)

Y = np_utils.to_categorical(y, 2)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3, random_state=0)

model = Sequential()
model.add(Dense(2, input_shape=(2,)))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
model.fit(Xtrain, Ytrain, batch_size=32, nb_epoch=50, validation_data=(Xtest, Ytest))

score = model.evaluate(Xtest, Ytest, verbose=0)
print("score: %.3f, accuracy: %.3f" % (score[0], score[1]))

Y_ = model.predict(X)
y_ = np_utils.categorical_probas_to_classes(Y_)
plot_dataset(X, y_)

X, y = read_dataset("../data/moons.csv")
y = y.astype("int")
print(X.shape, y.shape)
plot_dataset(X, y)

Y = np_utils.to_categorical(y, 2)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3, random_state=0)

model = Sequential()
model.add(Dense(50, input_shape=(2,)))
model.add(Activation("relu"))
model.add(Dense(2))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
model.fit(Xtrain, Ytrain, batch_size=32, nb_epoch=50, validation_data=(Xtest, Ytest))

score = model.evaluate(Xtest, Ytest, verbose=0)
print("score: %.3f, accuracy: %.3f" % (score[0], score[1]))

Y_ = model.predict(X)
y_ = np_utils.categorical_probas_to_classes(Y_)
plot_dataset(X, y_)

model = Sequential()
model.add(Dense(50, input_shape=(2,)))
model.add(Activation("relu"))
model.add(Dense(100))
model.add(Activation("relu"))
model.add(Dense(2))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
model.fit(Xtrain, Ytrain, batch_size=32, nb_epoch=50, validation_data=(Xtest, Ytest))

score = model.evaluate(Xtest, Ytest, verbose=0)
print("score: %.3f, accuracy: %.3f" % (score[0], score[1]))

Y_ = model.predict(X)
y_ = np_utils.categorical_probas_to_classes(Y_)
plot_dataset(X, y_)

X, y = read_dataset("../data/saturn.csv")
y = y.astype("int")
print(X.shape, y.shape)
plot_dataset(X, y)

model = Sequential()
model.add(Dense(50, input_shape=(2,)))
model.add(Activation("relu"))
model.add(Dense(100))
model.add(Activation("relu"))
model.add(Dense(2))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
model.fit(Xtrain, Ytrain, batch_size=32, nb_epoch=50, validation_data=(Xtest, Ytest))

score = model.evaluate(Xtest, Ytest, verbose=0)
print("score: %.3f, accuracy: %.3f" % (score[0], score[1]))

Y_ = model.predict(X)
y_ = np_utils.categorical_probas_to_classes(Y_)
plot_dataset(X, y_)

model = Sequential()

model.add(Dense(1024, input_shape=(2,)))
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(Dense(128))
model.add(Activation("relu"))

model.add(Dense(2))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
model.fit(Xtrain, Ytrain, batch_size=32, nb_epoch=50, validation_data=(Xtest, Ytest))

score = model.evaluate(Xtest, Ytest, verbose=0)
print("score: %.3f, accuracy: %.3f" % (score[0], score[1]))

Y_ = model.predict(X)
y_ = np_utils.categorical_probas_to_classes(Y_)
plot_dataset(X, y_)



