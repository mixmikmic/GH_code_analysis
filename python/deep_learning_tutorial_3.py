import keras
import numpy

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.datasets import mnist

from keras.callbacks import TensorBoard

batch_size = 128
num_classes = 10
TRAIN_EXAMPLES = 60000
TEST_EXAMPLES = 10000

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape the dataset to convert the examples from 2D matrixes to 1D arrays.
x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)

# to make quick runs, select a smaller set of images.
train_mask = numpy.random.choice(x_train.shape[0], TRAIN_EXAMPLES, replace=False)
x_train = x_train[train_mask, :].astype('float32')
y_train = y_train[train_mask]
test_mask = numpy.random.choice(x_test.shape[0], TEST_EXAMPLES, replace=False)
x_test = x_test[test_mask, :].astype('float32')
y_test = y_test[test_mask]

# normalize the input
x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy', 'mae'])

EXPERIMENT_COUNTER = 1

tensorboard = TensorBoard(log_dir='./logs/experiment-{}'.format(EXPERIMENT_COUNTER), histogram_freq=0,
                          write_graph=False, write_images=False)
epochs = 10
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=0,
                    validation_data=(x_test, y_test),
                    callbacks=[tensorboard])
EXPERIMENT_COUNTER += 1



