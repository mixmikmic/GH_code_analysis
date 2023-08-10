import numpy as np

x = np.array(([3,5],[5,1],[10,2]), dtype = float)

y = np.array(([75],[82],[93]), dtype = float)

x

y

# Scaling inputs and output
x = x/np.amax(x, axis=0) 
# Here, axis=0 means take the max of each column and devide each element of that column by the max value

x

y = y/100 # Max test score is 100.

y

from IPython.display import Image
Image(filename="nn1.png")

Image(filename="nn2.png") 
#Screenshot from welch labs tutorial

Image(filename="nn3.png") 

Image(filename="nn4.png") 

Image(filename="nn5.png") 

Image(filename="nn61.png") 

Image(filename="nn7.png") 

Image(filename="nn8.png") 

Image(filename="nn9.png") 

Image(filename="nn10.png") 

Image(filename="nn11.png") 

# Demonstartion of sigmoid function, not related to nn coding
def sigmoid(z):
        #Apply sigmoid function
        return 1/(1+np.exp(-z))

testInput = np.arange(-6,6,0.01)
import matplotlib.pyplot as plt
plt.plot(testInput, sigmoid(testInput), linewidth=2)
plt.grid(1)
plt.show()

sigmoid(1)

sigmoid(np.array([-1,0,1]))

sigmoid(np.random.randn(3,3))

class Neural_Network(object):
    def __init__(self):
        # Define Hyperparameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3
        
        # Weights(Parameters)
        self.w1 = np.random.rand(self.inputLayerSize,                                  self.hiddenLayerSize)
        self.w2 = np.random.rand(self.hiddenLayerSize,                                  self.outputLayerSize)
        
    def forward(self, x):
        # propagate inputs through the network
        self.z2 = np.dot(x, self.w1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.w2)
        yHat = self.sigmoid(self.z3)
        return yHat
    
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar or vector or matrix
        return 1/(1+np.exp(-z))
    

# Initialize the neural network. Call the forward function and predict yHat from NN.
NN = Neural_Network()
yHat = NN.forward(x)

yHat

y

Image(filename="nn12.png") 

import time

weightsToTry = np.linspace(-5, 5, 1000)
costs = np.zeros(1000)

startTime = time.clock()
for i in range(1000):
    NN.w1[0,0] = weightsToTry[i]
    yHat = NN.forward(x)
    costs[i] = 0.5*sum((y-yHat)**2)

endTime = time.clock()

timeElapsed = endTime-startTime
timeElapsed

Image(filename="nn13.png") 

#from videoSupport import *
import time

weightsToTry = np.linspace(-5,5,1000)
costs = np.zeros((1000, 1000))

startTime = time.clock()
for i in range(1000):
    for j in range(1000):
        NN.w1[0,0] = weightsToTry[i]
        NN.w1[0,1] = weightsToTry[j]
        yHat = NN.forward(x)
        costs[i] = 0.5*sum((y-yHat)**2)

endTime = time.clock()

timeElapsed = endTime-startTime
timeElapsed

#We have below five equation

Image(filename="nn141.png") 

Image(filename="nn15.png") 

Image(filename="nn16.png") 

Image(filename="nn17.png") 

Image(filename="nn18.png") 

Image(filename="nn191.png") 



