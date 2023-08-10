from sklearn import datasets
from sklearn.datasets import fetch_mldata
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, MaxPooling2D
from keras.datasets import mnist

from neural_network import NeuralNetworkImpl
from utils import compute_accuracy_multilabel

# This notebook could be run with one of two datasets. By default uses the much larger MNIST dataset
# with 60,000 train examples and 10,000 test examples.
dataset_to_use = 'mnist'

if dataset_to_use == 'digits':
# SKLEARN digits dataset
    digits = datasets.load_digits()
    X = digits['data'].T
    Y = digits['target']
    print(X.shape)
    print(Y.shape)
    train_size = 1500
    test_size = X.shape[1] - train_size

    X_train_images = digits.images[0:train_size]
    X_test_images = digits.images[train_size:]

    X_train = X[:, 0:train_size]
    Y_train_labels = Y[0:train_size]
    X_test = X[:, train_size:]
    Y_test_labels = Y[train_size:]

    Y_train = np.zeros((10, Y_train_labels.shape[0]))
    Y_train[Y_train_labels[:], np.arange(Y_train_labels.shape[0])] = 1

    Y_test = np.zeros((10, Y_test_labels.shape[0]))
    Y_test[Y_test_labels[:], np.arange(Y_test_labels.shape[0])] = 1
elif dataset_to_use == 'mnist':
# MNIST dataset
    (X_train_images, Y_train_labels), (X_test_images, Y_test_labels) = mnist.load_data()

    # Normalize the image data because the data range is [0, 255]. Without normalization the
    # neural net implementation performs quite miserable, usually always predicting just one 
    # label for all inputs.
    X_train_images = X_train_images / 255.
    X_test_images = X_test_images / 255.
    
    X_train = X_train_images.reshape((X_train_images.shape[0], -1)).T
    X_test = X_test_images.reshape((X_test_images.shape[0], -1)).T
    
    X_train = X_train
    X_test = X_test

    Y_train = np.zeros((10, Y_train_labels.shape[0]))
    Y_train[Y_train_labels[:], np.arange(Y_train_labels.shape[0])] = 1

    Y_test = np.zeros((10, Y_test_labels.shape[0]))
    Y_test[Y_test_labels[:], np.arange(Y_test_labels.shape[0])] = 1
else:
    raise "No dataset chosen"

# Plot some training examples
images_and_labels = list(zip(X_train_images, Y_train_labels))
for index, (image, label) in enumerate(images_and_labels[:12]):
    plt.subplot(3, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)
plt.show()

# Train a neural network using my implementation with two hidden layers of sizes 10 and 5 and one output 
# layer using softmax activation and 10 outputs, one per label.
model_nn = NeuralNetworkImpl(layer_sizes=[10, 5, 10], 
                             layer_activations=['relu', 'relu', 'softmax'], 
                             alpha=0.001, 
                             epochs=300, 
                             mini_batch_size=64, 
                             regularization=0.1, 
                             optimization_algorithm='adam')
model_nn.train(X_train, Y_train)
accuracy = compute_accuracy_multilabel(model_nn, X_train, Y_train)
print("Neural net train accuracy: {}".format(accuracy)) 
accuracy = compute_accuracy_multilabel(model_nn, X_test, Y_test)
print("Neural net test accuracy: {}".format(accuracy))

# Plot test examples in which the neural net was wrong.
expected = Y_test
predicted = model_nn.predict(X_test)

images_and_predictions = list(zip(X_test_images, np.argmax(predicted, axis=0), np.argmax(expected, axis=0)))
filtered_examples = [triplet for triplet in images_and_predictions if triplet[1] != triplet[2]]
for index, (image, prediction, expected) in enumerate(filtered_examples[:12]):
    plt.subplot(2, 6, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title("P:{},E:{}".format(prediction, expected))

plt.show()

# Plot test examples in which the neural net was right.
filtered_examples = [triplet for triplet in images_and_predictions if triplet[1] == triplet[2]]
for index, (image, prediction, expected) in enumerate(filtered_examples[:12]):
    plt.subplot(2, 6, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title("P:{},E:{}".format(prediction, expected))

plt.show()

# Train a Keras neural net similar to the one used above to compare my implementation with the one in Keras.
model = Sequential()
model.add(Dense(10, input_dim=28*28))
model.add(Activation('relu'))
model.add(Dense(5))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train.T, Y_train.T, epochs=300, batch_size=64, verbose=1)

score = model.evaluate(X_train.T, Y_train.T, batch_size=64)
print("Keras NN score: {}".format(score))

score = model.evaluate(X_test.T, Y_test.T, batch_size=64)
print("Keras NN score: {}".format(score))

# Train another neural net using my implementation, this time using bigger hidden layers, 
# just to see if that helps. The number of params to train is roughly twice bigger now.
model_nn_2 = NeuralNetworkImpl(layer_sizes=[20, 10, 10], 
                             layer_activations=['relu', 'relu', 'softmax'], 
                             alpha=0.001, 
                             epochs=300, 
                             mini_batch_size=64, 
                             regularization=0.1, 
                             optimization_algorithm='adam')
model_nn_2.train(X_train, Y_train)
accuracy = compute_accuracy_multilabel(model_nn_2, X_train, Y_train)
print("Neural net train accuracy: {}".format(accuracy)) 
accuracy = compute_accuracy_multilabel(model_nn_2, X_test, Y_test)
print("Neural net test accuracy: {}".format(accuracy))

# Plot test examples in which the neural net was wrong.
expected = Y_test
predicted = model_nn_2.predict(X_test)

images_and_predictions = list(zip(X_test_images, np.argmax(predicted, axis=0), np.argmax(expected, axis=0)))
filtered_examples = [triplet for triplet in images_and_predictions if triplet[1] != triplet[2]]
for index, (image, prediction, expected) in enumerate(filtered_examples[:30]):
    plt.subplot(5, 6, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title("P:{},E:{}".format(prediction, expected))

plt.show()

# An even bigger neural net, which has more params to train than the training inputs. 
# Just curious to see if that helps.
model_nn_3 = NeuralNetworkImpl(layer_sizes=[100, 100, 10], 
                             layer_activations=['relu', 'relu', 'softmax'], 
                             alpha=0.001, 
                             epochs=300, 
                             mini_batch_size=64, 
                             regularization=0.1, 
                             optimization_algorithm='adam')
model_nn_3.train(X_train, Y_train)
accuracy = compute_accuracy_multilabel(model_nn_3, X_train, Y_train)
print("Neural net train accuracy: {}".format(accuracy)) 
accuracy = compute_accuracy_multilabel(model_nn_3, X_test, Y_test)
print("Neural net test accuracy: {}".format(accuracy))

X_input = Input((28, 28, 1))

# zero padding
X = ZeroPadding2D((3, 3))(X_input)

# first conv layer, patch is 7x7, stride of 1, 32 filters
X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
# apply batch norm after the layer
X = BatchNormalization(axis = 3, name = 'bn0')(X)
X = Activation('relu')(X)

# max pooling
X = MaxPooling2D((2, 2), name='max_pool0')(X)

# second conv layer, smaller patch than the first one and with more filters: 64
X = Conv2D(64, (5, 5), strides = (1, 1), name = 'conv1')(X)
X = BatchNormalization(axis = 3, name = 'bn1')(X)
X = Activation('relu')(X)

# max pooling
X = MaxPooling2D((2, 2), name='max_pool1')(X)

# convert the outputs to vectors and feed them to a fully connected layer, 
# which produces 10 outputs using softmax activation
X = Flatten()(X)
X = Dense(10, activation='softmax', name='fc')(X)

model = Model(inputs = X_input, outputs = X, name='mnist_conv_net')

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

X_train_conv = X_train_images.reshape((60000, 28, 28 ,1))
X_test_conv = X_test_images.reshape((10000, 28, 28 ,1))

model.fit(X_train_conv, Y_train.T, epochs=10, batch_size=64, verbose=1)

score = model.evaluate(X_train_conv, Y_train.T, batch_size=64)
print("Keras NN score: {}".format(score))

score = model.evaluate(X_test_conv, Y_test.T, batch_size=64)
print("Keras NN score: {}".format(score))

# Plot test examples in which the conv net was wrong.
expected = Y_test
predicted = model.predict(X_test_conv).T

images_and_predictions = list(zip(X_test_images, np.argmax(predicted, axis=0), np.argmax(expected, axis=0)))
filtered_examples = [triplet for triplet in images_and_predictions if triplet[1] != triplet[2]]
for index, (image, prediction, expected) in enumerate(filtered_examples[:24]):
    plt.subplot(4, 6, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title("P:{},E:{}".format(prediction, expected))

plt.show()

model.summary()



