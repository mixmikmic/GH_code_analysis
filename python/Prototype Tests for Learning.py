get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib inline')

from fingerprint import GraphFingerprint
from wb import WeightsAndBiases
from itertools import combinations
from random import choice, sample
from numpy.random import permutation
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split, ShuffleSplit, cross_val_score
from sklearn.preprocessing import LabelBinarizer
from autograd import grad
from time import time

import autograd.numpy as np
import networkx as nx
import math
import matplotlib.pyplot as plt
from numba import jit

shapes = dict()
shapes[0] = 10
shapes[1] = 10
shapes[2] = 10
wb = WeightsAndBiases(2, shapes)
# wb[0]

def make_random_graph(nodes, n_edges, features_dict):
    """
    Makes a randomly connected graph. 
    """
    
    G = nx.Graph()
    for n in nodes:
        G.add_node(n, features=features_dict[n])
    
    for i in range(n_edges):
        u, v = sample(G.nodes(), 2)
        G.add_edge(u, v)
        
    return G

# features_dict will look like this:
# {0: array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
#  1: array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
#  2: array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
#  3: array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
#  4: array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
#  5: array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
#  6: array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
#  7: array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
#  8: array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
#  9: array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])}

all_nodes = [i for i in range(10)]    
lb = LabelBinarizer()
features_dict = {i:lb.fit_transform(all_nodes)[i] for i in all_nodes}

G = make_random_graph(sample(all_nodes, 6), 5, features_dict)
G.edges(data=True)
# G.nodes(data=True)

def score(G):
    """
    The regressable score for each graph will be the sum of the 
    (square root of each node + the sum of its neighbors.)
    """
    sum_score = 0
    for n, d in G.nodes(data=True):
        sum_score += math.sqrt(n)
        
        for nbr in G.neighbors(n):
            sum_score += nbr
    return sum_score

score(G)

syngraphs = [make_random_graph(sample(all_nodes, 6), 5, features_dict) for i in range(1000)]

len(syngraphs)

fingerprints = np.zeros((len(syngraphs), 10))

for i, g in enumerate(syngraphs):
    gfp = GraphFingerprint(g, 2, shapes)
    fp = gfp.compute_fingerprint(wb.vect, wb.unflattener)
    fingerprints[i] = fp

import pandas as pd
X = pd.DataFrame(np.array(fingerprints))
Y = [score(g) for g in syngraphs]
Y

# A simple test - the weights are random, so given the random weights, what is the prediction accuracy using
# random forest?

cv = ShuffleSplit(n=len(X), n_iter=10)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

rfr = RandomForestRegressor()
rfr.fit(X_train, Y_train)
# preds = np.rint(rfr.predict(X_test))
preds = rfr.predict(X_test)

from sklearn.metrics import mean_squared_error as mse

print(preds)
mse(preds, Y_test)

# How does this compare with randomly shuffled data?
mse(permutation(Y_test), Y_test)

[i for i in zip(Y_test, preds)]

def predict(wb_vect, wb_unflattener, graph_fp):#, linweights):
    """
    Given the weights and biases for each layer, make a prediction for the graph.
    """
    fp = graph_fp.compute_fingerprint(wb_vect, wb_unflattener)
    wb = wb_unflattener(wb_vect)
    top_layer = max(wb.keys())
    linweights = wb[top_layer]['linweights']
    return np.dot(fp, linweights)

predict(wb.vect, wb.unflattener, gfp)

@jit
def train_loss(wb_vect, wb_unflattener):
    """
    Training loss function - should take in a vector.
    """
    sum_loss = 0
    for i, g in enumerate(syngraphs):
        gfp = GraphFingerprint(g, 2, shapes)
        pred = predict(wb_vect, wb_unflattener, gfp)
        loss = len(g.nodes()) - predict(wb_vect, wb_unflattener, gfp)
        sum_loss = sum_loss + loss ** 2
    
    return sum_loss / len(syngraphs)

train_loss(wb.vect, wb.unflattener)

def sgd(grad, wb_vect, wb_unflattener, callback=None, num_iters=200, step_size=0.1, mass=0.9):
    """
    Stochastic gradient descent with momentum.
    """
    velocity = np.zeros(len(wb_vect))
    for i in range(num_iters):
        print(i)
        g = grad(wb_vect, wb_unflattener)
        # if callback: callback(x, i, g)
        velocity = mass * velocity - (1.0 - mass) * g
        wb_vect += step_size * velocity
        print(train_loss(wb_vect, wb_unflattener))
    return wb_vect

train_loss(wb.vect, wb.unflattener)

grad_func = grad(train_loss)

sgd(grad_func, wb.vect, wb.unflattener, num_iters=200)

trained_weights = wb.unflattener(wb.vect)[2]['linweights']
trained_weights

test_graphs = [make_random_graph(sample(all_nodes, 6), 5, features_dict) for i in range(100)]

test_fingerprints = np.zeros((len(test_graphs), 10))
# test_fingerprints
for i, g in enumerate(test_graphs):
    gfp = GraphFingerprint(g, 2, shapes)
    fp = gfp.compute_fingerprint(wb.vect, wb.unflattener)
    test_fingerprints[i] = fp

# test_fingerprints

preds = []
for i, g in enumerate(test_graphs):
    gfp = GraphFingerprint(g, 2, shapes)
#     fp = gfp.compute_fingerprint(wb.vect, wb.unflattener)
    preds.append(predict(wb.vect, wb.unflattener, gfp)[0])
# preds[0]

Y_test = [score(g) for g in syngraphs]

[i for i in zip(Y_test, preds)]



plt.scatter(preds, n_nodes, alpha=0.3)
plt.xlabel('predictions')
plt.ylabel('actual')
plt.title('number of nodes')

class Class(object):
    """docstring for ClassName"""
    def __init__(self, arg):
        super(Class, self).__init__()
        self.arg = arg
        
    def __iter__():
        pass
        
    def function(self, value, other_thing):
        return value['k']['v']['x'] ** 2 + value['y'] ** 3
    
    def function2(self, value):
        return np.sum(np.dot(value['arr1'], value['arr2'])) + 1
        
        
# def function(value):
#     return value ** 2

c = Class(np.random.random((10,10)))

from collections import OrderedDict
value = dict({'k':{'v':{'x':3.0}}, 'y':2.0})
gradfunc = grad(c.function)
gradfunc(value, 'string')

def fun2(value):
    return np.sum(np.dot(value['arr1'], value['arr2']))

value = {'arr1':np.random.random((10,10)), 'arr2':np.random.random((10,10))}
gradfunc = grad(fun2)(value)
gradfunc

value = {'arr1':np.random.random((10,10)), 'arr2':np.random.random((10,10))}
# value
gradfunc = grad(c.function2)
gradfunc(value)
# np.dot(c.arg, value['arr1'])# , c.arg)
# c.function2(value)

np.dot(value['arr1'], value['arr2'])





