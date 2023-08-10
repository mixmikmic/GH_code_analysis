from conx import Network, Layer, SGD

#net = Network("XOR Network", 2, 4, 1, activation="sigmoid")

net = Network("XOR Network")
net.add(Layer("input", shape=2))
net.add(Layer("hidden", shape=4, activation='sigmoid'))
net.add(Layer("output", shape=1, activation='sigmoid'))
net.connect()

dataset = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0])
]
net.compile(error='mean_squared_error', optimizer=SGD(lr=0.3, momentum=0.9))
net.dataset.load(dataset)

net.reset(seed=3863479522)
net.train(epochs=2000, accuracy=1, report_rate=25, plot=True)

import numpy as np

def sigmoid(x):
    return 1/(np.exp(-x)+1)

def my_compute_activations(vector, layer, net):
    weights, biases = net[layer].keras_layer.get_weights()
    activations = []
    for j in range(len(biases)):
        sum = 0
        for i in range(len(vector)):
            sum += (vector[i] * weights[i][j])
        a = sigmoid(sum + biases[j])
        activations.append(a)
    return activations

def my_propagate(vector, net):
    for layer in ["hidden", "output"]:
        vector = my_compute_activations(vector, layer, net)
    return vector

dataset

for i in range(4):
    print(my_propagate(dataset[i][0], net), dataset[i][1])

for i in range(4):
    print(net.propagate(dataset[i][0]), dataset[i][1])

from conx import Network, Layer, SGD

#net = Network("XOR Network", 2, 4, 1, activation="sigmoid")

net = Network("XOR Network")
net.add(Layer("input", shape=2))
net.add(Layer("hidden1", shape=4, activation='sigmoid'))
net.add(Layer("hidden2", shape=2, activation='sigmoid'))
net.add(Layer("output", shape=1, activation='sigmoid'))
net.connect()

dataset = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0])
]
net.compile(error='mean_squared_error', optimizer=SGD(lr=0.3, momentum=0.9))
net.dataset.load(dataset)

net.reset(seed=3863479522)
net.train(epochs=2000, accuracy=1, report_rate=25, plot=True)

def my_propagate(vector, net):
    for layer in ["hidden1", "hidden2", "output"]:
        vector = output_val(vector, layer, net)
    return vector

my_propagate([0, 1], net)

net.propagate([0,1])

my_compute_activations([0, 1], "hidden1", net)

h = net.propagate_from("input", [0, 1], "hidden1")
h

net.propagate_from("hidden1", h, "output")



