import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import plot

get_ipython().magic('pylab inline')

class Neural_Network(object):
    def __init__(self, inputLayerSize=2, outputLayerSize=1, hiddenLayerSize=16, alpha=0):        
        #Define Hyperparameters
        self.inputLayerSize = inputLayerSize
        self.outputLayerSize = outputLayerSize
        self.hiddenLayerSize = hiddenLayerSize
        
        #Weights (parameters)
        self.W1 = np.random.random((self.inputLayerSize,self.hiddenLayerSize))*2 - 1
        self.W2 = np.random.random((self.hiddenLayerSize,self.outputLayerSize))*2 - 1
        
        self.W_h = 2*np.random.random((self.hiddenLayerSize,self.hiddenLayerSize)) - 1
        
        self.alpha = alpha
        
        
    def forward(self, X, prev_hidden=None):
        #Propogate inputs though network
        self.z2 = np.dot(X, self.W1)
        if prev_hidden is not None:
            self.z2 += np.dot(prev_hidden, self.W_h)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        self.a3 = self.sigmoid(self.z3) 
        return self.a3
        
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    # convert output of sigmoid function to its derivative
    def sigmoid_output_to_derivative(self, output):
        return output*(1-output)

import copy
np.random.seed(0)

# training dataset generation
int2binary = {}
binary_dim = 8

# this is just a dictionary mapping from ints to their binary representation.
# makes conversion later easier
largest_number = pow(2,binary_dim)
binary = np.unpackbits(
    np.array([range(largest_number)],dtype=np.uint8).T,axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]

# input variables
alpha = 0.1
input_dim = 2
hidden_dim = 16
output_dim = 1

# initialize neural network weights
nn = Neural_Network(inputLayerSize=input_dim, outputLayerSize=output_dim, hiddenLayerSize=hidden_dim, alpha=0.1)

W1_update = np.zeros_like(nn.W1)
W2_update = np.zeros_like(nn.W2)
W_h_update = np.zeros_like(nn.W_h)

# training logic. Train our RNN on 10,000 addition problems
for j in range(10000):
    
    # generate a simple addition problem (a + b = c)
    a_int = np.random.randint(largest_number/2) # int version
    a = int2binary[a_int] # binary encoding

    b_int = np.random.randint(largest_number/2) # int version
    b = int2binary[b_int] # binary encoding

    # true answer
    c_int = a_int + b_int
    c = int2binary[c_int]
    
    # where we'll store our best guess (binary encoded)
    d = np.zeros_like(c)

    overallError = 0
    
    output_layer_deltas = []
    hidden_layer_values = []
    # assume the hidden layer was zero to begin with so our NN can reference
    # the previous hidden layer
    hidden_layer_values.append(np.zeros(nn.hiddenLayerSize))
    
    # This is our forward propagation. Within each addition problem, we iterate over the two numbers,
    # feeding individual bits as inputs to our neural network. The idea is that because the network
    # is recurrent, it can remember what it has seen before, and will learn how to "carry the one"
    for position in range(binary_dim):
        
        # generate input and output
        X = np.array([[a[binary_dim - position - 1],
                     b[binary_dim - position - 1]]])
        y = np.array([[c[binary_dim - position - 1]]]).T
        
        # generate a prediction, feeding the previous hidden layer to the current one
        prediction = nn.forward(X, prev_hidden=hidden_layer_values[-1])

        # did we miss?... if so, by how much?
        output_error = y - prediction
        # this is the backpropagating error, represented by a delta.
        output_layer_deltas.append((output_error)*nn.sigmoidPrime(nn.z3))
        overallError += np.abs(output_error[0])
    
        # decode estimate so we can print it out
        d[binary_dim - position - 1] = np.round(prediction[0][0])
        
        # store hidden layer so we can use it in the next timestep
        hidden_layer_values.append(copy.deepcopy(nn.a2))
    
    future_layer_1_delta = np.zeros(nn.hiddenLayerSize)
    
    # Here is backward propagation. For each bit in the binary numbers, we compute the gradients of the
    # cost function with respect to our three weight matrices. We keep track of all the gradients, and
    # then update the weights for the next training step.
    for position in range(binary_dim):
        
        X = np.array([[a[position],b[position]]])
        hidden_layer = hidden_layer_values[-position-1]
        prev_hidden_layer = hidden_layer_values[-position-2]
        
        # error at output layer
        layer_2_delta = output_layer_deltas[-position-1]
        # error at hidden layer
        layer_1_delta = (future_layer_1_delta.dot(
                             nn.W_h.T) +layer_2_delta.dot(
                             nn.W2.T)) * nn.sigmoid_output_to_derivative(
        hidden_layer)

        # let's update all our weights so we can try again
        W2_update += np.atleast_2d(hidden_layer).T.dot(layer_2_delta)
        W_h_update += np.atleast_2d(prev_hidden_layer).T.dot(layer_1_delta)
        W1_update += X.T.dot(layer_1_delta)
        
        future_layer_1_delta = layer_1_delta

    # update the weights
    nn.W1 += W1_update * nn.alpha
    nn.W2 += W2_update * nn.alpha
    nn.W_h += W_h_update * nn.alpha
    
    W1_update *= 0
    W2_update *= 0
    W_h_update *= 0
    
    # print out progress
    if(j % 1000 == 0):
        print "Error:" + str(overallError)
        print "Pred:" + str(d)
        print "True:" + str(c)
        out = 0
        for index,x in enumerate(reversed(d)):
            out += x*pow(2,index)
        print str(a_int) + " + " + str(b_int) + " = " + str(out)
        print "------------"



