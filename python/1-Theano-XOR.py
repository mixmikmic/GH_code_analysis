from __future__ import print_function

import theano
import theano.tensor as T
import numpy as np
import time

X = theano.shared(value=np.asarray([[0, 1], [1, 0], [0, 0], [1, 1]]), name='X')
y = theano.shared(value=np.asarray([[0], [0], [1], [1]]), name='y')
print('X: {}\ny: {}'.format(X.get_value(), y.get_value()))

np.random.seed(42)
rng = np.random.RandomState(1234)

def layer(*shape):
    assert len(shape) == 2
    mag = 4. * np.sqrt(6. / sum(shape))
    W_value = np.asarray(rng.uniform(low=-mag, high=mag, size=shape), dtype=theano.config.floatX)
    b_value = np.asarray(np.zeros(shape[1], dtype=theano.config.floatX), dtype=theano.config.floatX)
    W = theano.shared(value=W_value, name='W_{}'.format(shape), borrow=True, strict=False)
    b = theano.shared(value=b_value, name='b_{}'.format(shape), borrow=True, strict=False)
    return W, b

W1, b1 = layer(2, 5)
W2, b2 = layer(5, 1)
print(W1.get_value())

output = T.nnet.sigmoid(T.dot(T.nnet.relu(T.dot(X, W1) + b1), W2) + b2) # The whole network
cost = T.mean((y - output) ** 2) # Mean squared error
updates = [(p, p - 0.1 * T.grad(cost, p)) for p in [W1, W2, b1, b2]] # Subgradient descent optimizer

train = theano.function(inputs=[], outputs=[], updates=updates)
test = theano.function(inputs=[], outputs=cost)

print('Cost before:', test())
start = time.time()
for i in range(10000):
    train()
end = time.time()
print('Cost after:', test())
print('Time (s):', end - start)

