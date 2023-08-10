import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from random import randint

def sigmoid(x):
    return 1./(1+np.exp(-x))

class NN:
    def __init__(self, nbr_inputs, nbr_outputs):
        self.nbr_outputs = nbr_outputs
        self.W = np.array([[0.]*nbr_inputs]*nbr_outputs) # weight[i,j] links input j to output i
        self.b = [0.]*nbr_outputs
        self.output = [0.]*nbr_outputs
        
    def forward(self, x):
        return sigmoid(np.dot(self.W,x) + self.b)
        
    def backprop(self, x, y, lr=0.01):
        self.output = self.forward(x)
        for iNeuron in range(self.nbr_outputs):
            grad_w = self.output[iNeuron] * (1. - self.output[iNeuron]) * (self.output[iNeuron] - y[iNeuron]) * x
            grad_b = self.output[iNeuron] * (1. - self.output[iNeuron]) * (self.output[iNeuron] - y[iNeuron])
            self.W[iNeuron] -= lr * grad_w
            self.b[iNeuron] -= lr * grad_b

nb_iter = 50000
X = []
Y = []
for i in range(nb_iter):
    x1 = randint(0,1)
    x2 = randint(0,1)
    X.append([x1,x2])
    Y.append([x1 & x2])
X = np.array(X, dtype=float)
Y = np.array(Y, dtype=float)

print(X)
print(Y)

model = NN(2,1)

for i in range(50000):
    model.backprop(X[i], Y[i], lr=0.01)

x = [1,1]
print("Weights : ", model.W)
print("Bias : ", model.b)
print()
print("Result : ", np.round(model.forward(x)[0]))
print("Probability : ", model.forward(x)[0])









