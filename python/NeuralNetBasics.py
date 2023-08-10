import numpy as np

# Create a "Sigmoid". This is the activation of a neuron.
# A function that will map any value to a value bettween 0 and 1
# Creates probabilities out of numbers
def nonlin(x,deriv=False):
    if(deriv==True):
        return (x*(1-x))
    return (1/(1+np.exp(-x)))

# Initialize the dataset as a matrix with input Data:
# Each row is a diferent training example
# Each column represents a diferent neuron 
X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])

# Output Data with one output neuron each
Y = np.array([[1],
              [0.7],
              [3],
              [0]])

# seed them to make them deterministic
# give random numbers with the same starting point (useful for debuging)
# so we can get the same sequence of generated numbers everytime we run the program
np.random.seed(1)

# Create synapse matrices.
# Initialize the weights of a neural network
# (it is a neural network with two layers of weights):
syn0 = 2 * np.random.random((3, 4)) - 1
syn1 = 2 * np.random.random((4, 1)) - 1

# Training code (loop)
for j in xrange(60000):
    # optimize the network for the given data set
    # First layer it's just our input data
    l0 = X
    # Prediction step
    # preform matrix multiplication bettween each layer and its synapse
    # Then run sigmoid function on the matrix to create the next layer
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))
    
    # With the above prediction of the output in l2 we can compare it to the expected 
    # output data using subtraction to get an error rate
    l2_error = Y - l2
    
    # Print the average error at a set interval to make sure it goes down every time
    if(j % 10000) == 0:
        print "Error:" + str(np.mean(np.abs(l2_error)))
        
    # Multiply the error rate by the slope of the sigmoid at the values in l2
    l2_delta = l2_error * nonlin(l2, deriv=True)
    
    # (Backpropagation) How much did l1 contributed to the error on l2 ?
    # Multiply layer 2 delta by synapse 1 transpose.
    l1_error = l2_delta.dot(syn1.T)
    
    # Get l1's delta by multlying it's error by the result of the sigmoid function
    l1_delta = l1_error * nonlin(l1, deriv=True)
    
    # (Gradient Descent) update weights 
    # Now that we have deltas for each of our layers, we can use them to update
    # our synapses rates to reduce the error rate more and more every iteration.
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
    
print "Output after"
print l2
print "Objective"
print Y

print l2



