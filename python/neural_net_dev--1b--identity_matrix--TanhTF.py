import matplotlib as mpl
mpl.use('Agg')
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy as sp
import seaborn as sns
import sys

sys.path.append('../code')

from NeuralNet import NeuralNet
from TransferFunctions import TanhTF

# columns are data points and rows are features
X = np.array([[1, 0], [1, 0], [0, 1]]).T
y = np.array([0, 0, 1])
d, N = np.shape(X)
C = np.unique(y).shape[0]

np.array([[-1, 0], [0,1]]).T

n = NeuralNet(X=X, y=y, hidden_nodes=2, 
              hiddenTF=TanhTF, outputTF=TanhTF,
              minibatch_size=3,
              summarise_frequency=3,
              eta0 = 0.1,
              convergence_delta=1e-4)

print(n.W1.shape)
n.W1 = np.array([[0.01161103, 0],[0, 1]])
print(n.W1.shape)
n.W1

print(n.W2.shape)
n.W2 = np.array([[-0.012, 0],[0, 1]])
print(n.W2.shape)
n.W2

n.__dict__

def point(vector, point_number):
    return vector[:, point_number-1:point_number]

point(X,1)

p = point(X,1)
print(p)
n.feed_forward_and_predict_Y(p)

point(n.Y,1)

n.Y

point(n.Y, 3)

p_num = 3
n.feed_forward(point(X, p_num))
dW1, dW2 = n.backprop(point(X, p_num), point(n.Y, p_num))
print(dW1)
print("")
print(dW2)

# send two training points in 
X[:, 1:3]

n.feed_forward_and_predict_Y(X[:, 1:3]) #, Y[:, 1:2])

dW1, dW2 = n.backprop(X[:, 1:3], n.Y[:, 1:3])
print(dW1)
print("")
print(dW2)

n.run(epochs=1)

n.results

n.feed_forward_and_predict_Y(X[:, 0:3]) #, Y[:, 0:3])

n.backprop(X[:, 0:3], n.Y[:, 0:3])

n.run(epochs=300)

n.results.tail()

sl = n.plot_square_loss()

l01 = n.plot_01_loss()

n.W1_tracking.tail(3)

f = n.plot_weight_evolution()

n.W1

n.W2

p1 = n.plot_sum_of_weights('W1')
p2 = n.plot_sum_of_weights('W2')

p1 = n.plot_norm_of_gradient(norm='W1')
p2 = n.plot_norm_of_gradient(norm='W1')

g1, g2 = n.gradients(n.X, n.Y)
print(g1)
print(g2)

np.linalg.norm(g1)

n.W1

n.W2

n.feed_forward(n.X)
dW1, dW2 = n.backprop(n.X, n.Y)
print(dW1)
print("")
print(dW2)

dW1_num, dW2_num = n.numerical_derivatives()
print(dW1)
print(dW2)

np.sum(np.abs(dW1 - dW1_num))

np.sum(np.abs(dW2 - dW2_num))

len(n.Y)

n.Y

np.arange(n.Y.shape[1])



