import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

from sklearn.datasets import make_classification
from sklearn.datasets import fetch_mldata
import numpy as np

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split

perceptron = NeuralNet(
    layers=[  # three layers: one hidden layer
        ('input', layers.InputLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape=(None, 2),     # 2 input pixels per batch
    output_nonlinearity=None,  # output layer uses identity function
    output_num_units=1,        # 1 target values [0, 1]

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=True,   # flag to indicate we're dealing with regression problem
    max_epochs=400,    # we want to train this many epochs
    verbose=1,
    )

print perceptron

X1, Y1 = make_classification(n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1)

perceptron.fit(X1.astype(np.float32), Y1.astype(np.uint16))

y_pred = perceptron.predict(X1.astype(np.float32))
print np.all(y_pred == Y1)

for n, pred in enumerate(y_pred):
    print u'Neural net prediction {}, expected {}, comparison {}.'.format(pred, Y1[n], np.abs(np.round(pred)) == Y1[n])

class0 = np.where(Y1==0)[0]
class1 = np.where(Y1==1)[0]

xx, yy = np.mgrid[-5:5:.01, -5:5:.01]
grid = np.c_[xx.ravel(), yy.ravel()]
probs = perceptron.predict_proba(grid.astype(np.float32)).reshape(xx.shape)

f, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu",
                      vmin=0, vmax=1)
ax_c = f.colorbar(contour)
ax_c.set_label("$P(y = 1)$")
ax_c.set_ticks([0, .25, .5, .75, 1])

ax.scatter(X1[class0, 0], X1[class0, 1], c='r')
ax.scatter(X1[class1, 0], X1[class1, 1], c='b')
plt.show()

import os
import gzip
import pickle
from urllib import urlretrieve

def pickle_load(f, encoding):
    return pickle.load(f)

DATA_URL = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
DATA_FILENAME = 'mnist.pkl.gz'

def _load_data(url=DATA_URL, filename=DATA_FILENAME):
    """Load data from `url` and store the result in `filename`."""
    if not os.path.exists(filename):
        print("Downloading MNIST dataset")
        urlretrieve(url, filename)

    with gzip.open(filename, 'rb') as f:
        return pickle_load(f, encoding='latin-1')

def load_data():
    """Get data with labels, split into training, validation and test set."""
    data = _load_data()
    X_train, y_train = data[0]
    X_valid, y_valid = data[1]
    X_test, y_test = data[2]
    y_train = np.asarray(y_train, dtype=np.int32)
    y_valid = np.asarray(y_valid, dtype=np.int32)
    y_test = np.asarray(y_test, dtype=np.int32)

    return dict(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test,
        num_examples_train=X_train.shape[0],
        num_examples_valid=X_valid.shape[0],
        num_examples_test=X_test.shape[0],
        input_dim=X_train.shape[1],
        output_dim=10,
    )

data = load_data()

net1 = NeuralNet(
        layers=[('input',  layers.InputLayer),
                ('hidden', layers.DenseLayer),
                ('output', layers.DenseLayer),
                ],
        # layer parameters:
        input_shape=(None, 28*28),
        hidden_num_units=100,           # number of units in 'hidden' layer
        output_nonlinearity=lasagne.nonlinearities.softmax,
        output_num_units=10,            # 10 target values for the digits 0, 1, 2, ..., 9

        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,

        max_epochs=10,
        verbose=1,
        )

plt.axis('off')
plt.imshow(data['X_train'][0, :].reshape(28, 28), cmap='gray')
print 'class', data['y_train'][0]

net1.fit(data['X_train'], data['y_train'])

idx = 24

print("Label: %s" % str(data['y_test'][idx]))
print("Predicted: %s" % str(net1.predict([data['X_test'][idx]])))

plt.axis('off')
plt.imshow(data['X_test'][idx].reshape(28, 28), cmap='gray')

get_ipython().magic("time y_pred = net1.predict(data['X_test'])")

print 'quantity of images on {}'.format(len(y_pred))

correct = 0
for n, pred in enumerate(y_pred):
    if str(pred) == str(data['y_test'][n]):
        correct = correct + 1    

print 'Corrected {}, {:.2f}%'.format(correct, 100 * float(correct)/len(data['y_test']))



