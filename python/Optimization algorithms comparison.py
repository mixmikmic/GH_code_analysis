import sklearn
import sklearn.datasets
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation

from neural_network import NeuralNetworkImpl
from utils import plot_decision_boundary, compute_accuracy

train_X, train_Y = sklearn.datasets.make_moons(n_samples=1000, noise=.4)
plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
plt.show()
train_X = train_X.T
train_Y = train_Y.reshape((1, train_Y.shape[0]))

model_nn_sgd = NeuralNetworkImpl(layer_sizes=[5, 3, 1],
    layer_activations=['relu', 'relu', 'sigmoid'], alpha=0.001,
    epochs=10000, optimization_algorithm='sgd')
model_nn_sgd.train(train_X, train_Y)

accuracy = compute_accuracy(model_nn_sgd, train_X, train_Y)
print("Neural net (sgd) model train accuracy: {}".format(accuracy))
plot_decision_boundary(model_nn_sgd, train_X, train_Y)

model_nn_mom = NeuralNetworkImpl(layer_sizes=[5, 3, 1],
    layer_activations=['relu', 'relu', 'sigmoid'], alpha=0.001,
    epochs=10000, mini_batch_size=64, optimization_algorithm='momentum')
model_nn_mom.train(train_X, train_Y)

accuracy = compute_accuracy(model_nn_mom, train_X, train_Y)
print("Neural net (momentum) model train accuracy: {}".format(accuracy))
plot_decision_boundary(model_nn_mom, train_X, train_Y)

model_nn_rmsprop = NeuralNetworkImpl(layer_sizes=[5, 3, 1],
    layer_activations=['relu', 'relu', 'sigmoid'], alpha=0.001,
    epochs=10000, mini_batch_size=64, optimization_algorithm='rmsprop')
model_nn_rmsprop.train(train_X, train_Y)

accuracy = compute_accuracy(model_nn_rmsprop, train_X, train_Y)
print("Neural net (rmsprop) model train accuracy: {}".format(accuracy))
plot_decision_boundary(model_nn_rmsprop, train_X, train_Y)

model_nn_adam = NeuralNetworkImpl(layer_sizes=[5, 3, 1],
    layer_activations=['relu', 'relu', 'sigmoid'], alpha=0.001,
    epochs=10000, mini_batch_size=64, optimization_algorithm='adam')
model_nn_adam.train(train_X, train_Y)

accuracy = compute_accuracy(model_nn_adam, train_X, train_Y)
print("Neural net (adam) model train accuracy: {}".format(accuracy))
plot_decision_boundary(model_nn_adam, train_X, train_Y)

# Compare the results with a Keras model using Adam
model = Sequential()
model.add(Dense(5, input_dim=2))
model.add(Activation('relu'))
model.add(Dense(3))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(train_X.T, np.squeeze(train_Y), epochs=10000, batch_size=64, verbose=0)
score = model.evaluate(train_X.T, np.squeeze(train_Y), batch_size=64)
print("\nKeras NN loss/accuracy:")
for label, value in zip(model.metrics_names, score):
    print("{}: {}".format(label, value))

