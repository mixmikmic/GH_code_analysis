from random import random, seed

def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    # Creating hidden layers according to the number of inputs
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    # Creating output layer according to the number of hidden layers
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network

# It is good practice to initialize the network weights to small random numbers. 
# In this case, will we use random numbers in the range of 0 to 1.
# To achieve that we seed random with 1
seed(1)

# 2 input units, 1 hidden unit and 2 output units
network = initialize_network(2, 1, 2)
# You can see the hidden layer has one neuron with 2 input weights plus the bias.
# The output layer has 2 neurons, each with 1 weight plus the bias.

for layer in network:
    print(layer)

# Implementation
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation

from math import exp

def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))

# Foward propagate is self-explanatory
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

inputs = [1, 0, None]
output = forward_propagate(network, inputs)

# Running the example propagates the input pattern [1, 0] and produces an output value that is printed.
# Because the output layer has two neurons, we get a list of two numbers as output.
output

# Calulates the derivation from an neuron output
def transfer_derivative(output):
    return output * (1.0 - output)

def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

expected = [0, 1]
backward_propagate_error(network, expected)
# delta: error value
for layer in network:
    print(layer)



