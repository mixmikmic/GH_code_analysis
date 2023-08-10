import numpy as np

# sigmoid function
# maps any value to a value between 0 and 1.
# use it to convert numbers to probabilities.
def sigmoid (x, deriv=False):
    # implement the gradient inside for convenience
    if(deriv):
        return x * (1-x)
    return 1 / (1 + np.exp(-x))

# Data 
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])
          
y = np.array([[0,0,1,1]]).T

# synapse zero:  weight matrix
# - Only one for one layer
# - Its dimension is (3,1) because we have 3 inputs and 1 output.
syn0 = 2*np.random.random((3,1)) - 1

for _ in range(10000):

    # make a prediction
    l0 = X
    l1 = sigmoid(np.dot(l0, syn0))
    # this returns a 4 by 1 vector with the expected values of our inputs

    # calculate a simple linear error
    l1_error = y - l1

    # multiply the error by the 
    # slope of the sigmoid at the values in l1
    # Because sigmoid is higher in 0 regien when we have low confidence
    # and we want to change those weight more heavily
    l1_delta = l1_error * sigmoid(l1, True)

    # update weights
    syn0 += np.dot(l0.T, l1_delta)

print ("Output After Training:\n{}".format(l1))

import numpy as np

def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
    
X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
                
y = np.array([[0],
              [1],
              [1],
              [0]])

# randomly initialize our weights with mean 0
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

for j in range(60000):

    # make a prediction
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))

    # calculate the error
    l2_error = y - l2
    
    # print the error every 10_000 steps to see if working
    if (j % 10000) == 0:
        print ("Error: {}".format(str(np.mean(np.abs(l2_error)))))
        
    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    l2_delta = l2_error*nonlin(l2,deriv=True)

    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dot(syn1.T)
    
    # in what direction is the target l1?
    l1_delta = l1_error * nonlin(l1,deriv=True)

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

print ("\nOutput After Training:\n{}".format(l2))



