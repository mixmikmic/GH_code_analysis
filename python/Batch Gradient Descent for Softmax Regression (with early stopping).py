import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

iris=load_iris()
X=iris['data']
y=iris['target']

X_with_bias = np.c_[np.ones([len(X), 1]), X]
np.random.seed(1234)

test_ratio = 0.2
validation_ratio = 0.2
total_size = len(X_with_bias)

test_size = int(total_size * test_ratio)
validation_size = int(total_size * validation_ratio)
train_size = total_size - test_size - validation_size

rnd_indices = np.random.permutation(total_size)

X_train = X_with_bias[rnd_indices[:train_size]]
y_train = y[rnd_indices[:train_size]]
X_valid = X_with_bias[rnd_indices[train_size:-test_size]]
y_valid = y[rnd_indices[train_size:-test_size]]
X_test = X_with_bias[rnd_indices[-test_size:]]
y_test = y[rnd_indices[-test_size:]]

def one_hot(Y):
    nclasses=Y.max()+1
    m = len(Y)
    Y_one_hot=np.zeros((m,nclasses))
    Y_one_hot[np.arange(m),Y]=1
    return Y_one_hot

y_valid[:10]

one_hot(y_valid[:10])

y_train_prob = one_hot(y_train)
y_valid_prob = one_hot(y_valid)
y_test_prob = one_hot(y_test)

def softmax(sk_X):
    top = np.exp(sk_X)
    bottom = np.sum(top,axis=1,keepdim=True)
    return top/bottom

n_inputs = X_train.shape[1]
n_outputs = len(np.unique(y_train))

print (n_inputs, n_outputs)



