import pickle
import time
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD

train_x = pickle.load(open("MNIST_train_x.pkl", 'rb'))
train_y = pickle.load(open("MNIST_train_y.pkl", 'rb'))
test_x = pickle.load(open("MNIST_test_x.pkl", 'rb'))
test_y = pickle.load(open("MNIST_test_y.pkl", 'rb'))
train_x_short = train_x[:20000]
train_y_short = train_y[:20000]

# Softmax output layer, mse
model = Sequential()
model.add(Dense(128, input_dim=784))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01)
model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
model.fit(train_x_short, train_y_short, batch_size=128, nb_epoch=10, validation_split=0.2, verbose=2)
print()

# Softmax output layer, categorical crossentropy
model = Sequential()
model.add(Dense(128, input_dim=784))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x_short, train_y_short, batch_size=128, nb_epoch=10, validation_split=0.2, verbose=2)

start = time.time()
model = Sequential()
model.add(Dense(128, input_dim=784))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x_short, train_y_short, batch_size=128, nb_epoch=10, validation_split=0.2, verbose=2)
end = time.time()
print("Model took  {} seconds to complete".format(end - start))

start = time.time()
model = Sequential()
model.add(Dense(128, input_dim=784))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x_short, train_y_short, batch_size=64, nb_epoch=7, validation_split=0.2, verbose=2)
end = time.time()
print("Model took  {} seconds to complete".format(end - start))

start = time.time()
model = Sequential()
model.add(Dense(128, input_dim=784))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x_short, train_y_short, batch_size=32, nb_epoch=6, validation_split=0.2, verbose=2)
end = time.time()
print("Model took  {} seconds to complete".format(end - start))

start = time.time()
model = Sequential()
model.add(Dense(128, input_dim=784))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x_short, train_y_short, batch_size=16, nb_epoch=6, validation_split=0.2, verbose=2)
end = time.time()
print("Model took  {} seconds to complete".format(end - start))

model = Sequential()
model.add(Dense(128, input_dim=784))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

sgd = SGD(lr=0.001)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x_short, train_y_short, batch_size=32, nb_epoch=10, validation_split=0.2, verbose=2)

model2 = Sequential()
model2.add(Dense(128, input_dim=784))
model2.add(Activation('sigmoid'))
model2.add(Dense(10))
model2.add(Activation('softmax'))

sgd = SGD(lr=0.01)
model2.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model2.fit(train_x_short, train_y_short, batch_size=32, nb_epoch=10, validation_split=0.2, verbose=2)

model3 = Sequential()
model3.add(Dense(512, input_dim=784))
model3.add(Activation('relu'))
model3.add(Dense(256))
model3.add(Activation('relu'))
model3.add(Dense(128))
model3.add(Activation('relu'))
model3.add(Dense(64))
model3.add(Activation('relu'))
model3.add(Dense(10))
model3.add(Activation('softmax'))

sgd = SGD(lr=0.001)
model3.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model3.fit(train_x_short, train_y_short, batch_size=32, nb_epoch=10, validation_split=0.2, verbose=2)

model3 = Sequential()
model3.add(Dense(512, input_dim=784))
model3.add(Activation('sigmoid'))
model3.add(Dense(256))
model3.add(Activation('sigmoid'))
model3.add(Dense(128))
model3.add(Activation('sigmoid'))
model3.add(Dense(64))
model3.add(Activation('sigmoid'))
model3.add(Dense(10))
model3.add(Activation('softmax'))

sgd = SGD(lr=0.01)
model3.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model3.fit(train_x_short, train_y_short, batch_size=32, nb_epoch=10, validation_split=0.2, verbose=2)

