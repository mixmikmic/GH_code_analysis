# Let's start by importing relevant packages :
import numpy as np
import os
import cPickle as pickle
import gzip
import subprocess

# Define a function to unzip the mnist data and load it
def get_mnist_data():
    # Unzip and load the data set
    f = gzip.open("../data/mnist.pkl.gz", "rb")
    train, val, test = pickle.load(f)
    f.close()
    return train, val, test

# Format the mnist target function
def format_mnist(y):
    """ Convert the 1D to 10D """

    # convert to 10 categeories
    y_new = np.zeros((10, 1))
    y_new[y] = 1
    return y_new
    
# Let's load and format the data for our neural network
train, test, valid = get_mnist_data()
training_inputs = [np.reshape(x, (784, 1)) for x in train[0]]
training_results = [format_mnist(y) for y in train[1]]
training_data = zip(training_inputs, training_results)
test_fm = []
for i in range(len(test[0])):
    test_fm.append((test[0][i], test[1][i]))
test = test_fm

# Lets check the dimensions of our data
print len(training_data), len(test)

# Neural Network code per se
# All documentation is inline


class ActivationFunction(object):

    """ Activation function class
    
    We define the activation function and its derivative.
    Only one choice : the sigmoid.
    Feel free to add more !
    
    """

    @staticmethod
    def sigmoid():
        return lambda x: 1. / (1 + np.exp(-x))

    @staticmethod
    def dsigmoid():
        return lambda x: 1. / (1 + np.exp(-x)) * (1  - 1. / (1 + np.exp(-x)))

class CostFunction(object):

    """ Cost function class
    
    We define one function : the mse
    Feel free to add your own (like the cross entropy)
    """

    @staticmethod
    def mse():
        return lambda y,a: 0.5 * np.power(y-a, 2)

    @staticmethod
    def dmse():
        return lambda y,a: a - y

class BasicNeuralNet(object):


    """
    
    Neural network class
    Implemented in python
    
    Main parts :
    
    __init__ = initialisation of the neural networks
    the weights and biases are randomly initialised
    
    _forward_pass() = operates the feedforward pass discussed above
    _backward_pass() = operates the backpropagation pass discussed above
    _update_gradient() = computes dCdw and dCdb for a mini batch as explained above
    
    fit_SGD() = fit the neural network to the data
                it loops over epochs, create mini batches of the data
                and minimises the cost function by gradient descent
    score() = evaluate the classification performance on specified test_data
    
    """
    def __init__(
            self,
            sizes,
            lmbda = 0,
            actfuncname="sigmoid",
            costfuncname="mse",
            verbose=False):
        self._nlayers = len(sizes)
        self._sizes = sizes
        self._lmbda = lmbda
        # Random initialisation of biases and weights.
        #For the weights, use gaussian with std = sqrt(# of weights connecting to a neuron)
        # So that by the CLT, their sum is gaussian with std = 1
        # Add [0] for clearer indexing
        self._biases = [np.array([0])] + [np.random.randn(size, 1)
                                          for size in self._sizes[1:]]
        self._weights = [np.array([0])] + [np.random.randn(self._sizes[i], self._sizes[i - 1])/                                            np.sqrt(self._sizes[i-1])
                                           for i in range(1, self._nlayers)]

        # Initialisation of z
        self._z = [np.array([0])] + [np.zeros((size, 1))
                                     for size in self._sizes[1:]]

        # Activation function
        self._actfuncname = actfuncname
        if self._actfuncname == "sigmoid":
            self._actfunc = ActivationFunction.sigmoid()
            self._dactfunc = ActivationFunction.dsigmoid()

        # Cost function
        self._costfuncname = costfuncname
        if self._costfuncname == "mse":
            self._costfunc = CostFunction.mse()
            self._dcostfunc = CostFunction.dmse()


    def _forward_pass(self, x):
        # Initialisation of activation matrix
        self._a = [x]
        for layer in range(1, self._nlayers):
            self._z[layer] = np.dot(self._weights[layer], self._a[layer-1])                 + self._biases[layer]
            a = self._actfunc(self._z[layer])
            self._a.append(a)

        # For scoring
        return self._a[-1]

    def _backward_pass(self, y):
        # Initialisation of error matrix
        delta_L = self._dcostfunc(y, self._a[-1])             * self._dactfunc(self._z[-1])
        self._delta = [delta_L]
        for layer in range(1, self._nlayers - 1)[::-1]:
            delta_l = np.dot(
                self._weights[layer + 1].T, self._delta[self._nlayers - layer -2]) \
                * self._dactfunc(self._z[layer])

            self._delta = [delta_l] + self._delta
        self._delta = [np.array([0])] + self._delta

    def _update_gradient(self, batch, n_training):

        n = len(batch)

        # Initialise derivative of cost wrt bias and weights
        dCdb = [np.array([0])] + [np.zeros((size,1)) for size in self._sizes[1:]]
        dCdw = [np.array([0])] + [np.zeros((self._sizes[i], self._sizes[i - 1]))
                                  for i in range(1, self._nlayers)]
        # Loop over batch
        for X, y in batch:
            self._forward_pass(X)
            self._backward_pass(y)

            # Loop over layers
            for layer in range(1, self._nlayers):
                dCdb[layer] += self._delta[layer]/float(n)
                dCdw[layer] += np.dot(self._delta[layer], self._a[layer - 1].T)/float(n) + self._lmbda * self._weights[layer]/float(n_training)

        return dCdb, dCdw

    def fit_SGD(self, training_data, learning_rate, batch_size, epochs, test_data = None):

        n_samples = len(training_data)
        # Loop over epochs
        for ep in range(epochs):

            #Shuffle data
            np.random.shuffle(training_data)

            for k in xrange(0, n_samples, batch_size):
                batch = training_data[k:k+batch_size]

                dCdb, dCdw = self._update_gradient(batch, n_samples)
                # Update bias and weights
                self._biases = [self._biases[layer] - learning_rate * dCdb[layer]
                           for layer in range(self._nlayers)]
                self._weights = [self._weights[layer] - learning_rate * dCdw[layer]
                                 for layer in range(self._nlayers)]
            print "Epoch %s:" %ep, self.score(test_data), "/", len(test_data)

    def score(self, test_data):
        """ Score """

        test_results = [(np.argmax(self._forward_pass(np.reshape(x, (len(x), 1)))), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

# Define the parameters of the neural network
d_NN = {"sizes": [784, 30, 10],
            "actfuncname": "sigmoid",
            "costfuncname": "mse",
            "batch_size": 10,
            "learning_rate": 3,
            "epochs": 30,
            "lambda":0,
            "verbose": True}

# Set the seed for reproducibility
np.random.seed(10)

# Create NN model and fit
NeuralNet = BasicNeuralNet(
        d_NN["sizes"],
        lmbda = d_NN["lambda"],
        actfuncname=d_NN["actfuncname"],
        costfuncname=d_NN["costfuncname"],
        verbose=d_NN["verbose"])

NeuralNet.fit_SGD(
    training_data,
    d_NN["learning_rate"],
    d_NN["batch_size"],
    d_NN["epochs"],
    test_data=test)

# This may take a while to train ...

