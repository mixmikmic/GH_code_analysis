import numpy as np

# The activation function, we will use the sigmoid
def sigmoid(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

# define learning rate
learning_rate = 0.4

# input dataset, note we add bias 1 to the input data
X = np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1],[1,0,0]])
X = np.concatenate((np.ones((len(X), 1)), X), axis = 1)

# output dataset           
y = np.array([[0,1,1,0,1]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
weights_0 = 2*np.random.random((4,1)) - 1

# train the network with 50000 iterations
for iter in xrange(50000):

    # forward propagation
    layer_0 = X
    layer_1_output = sigmoid(np.dot(layer_0,weights_0))

    # how much difference? This will be the error of 
    # our estimation and the true value
    layer1_error = y - layer_1_output

    # multiply how much we missed by the
    # slope of the sigmoid at the values at output layer
    # we also multiply the input to take care of the negative case
    layer1_delta = learning_rate * layer1_error * sigmoid(layer_1_output,True)
    layer1_delta = np.dot(layer_0.T,layer1_delta)

    # update weights by simply adding the delta
    weights_0 += layer1_delta
    
print "Output After Training:"
print layer_1_output

