get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib.image as imgplot
import cPickle as pickle
import gzip
with gzip.open('mnist_4000.pkl.gz', 'rb') as f:
    (X,y) = pickle.load(f)
PIXELS = len(X[0,0,0,:])
X.shape, y.shape, PIXELS

#from create_mnist import load_data_2d
#X,y,PIXELS = load_data_2d('/home/dueo/dl-playground/data/mnist.pkl.gz')
#X.shape, y

from lasagne import layers
from lasagne import nonlinearities
from nolearn.lasagne import NeuralNet

net1 = NeuralNet(
    # Geometry of the network
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('hidden4', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, PIXELS, PIXELS), #None in the first axis indicates that the batch size can be set later
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2), #pool_size used to be called ds in old versions of lasagne
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    hidden4_num_units=500,
    output_num_units=10, output_nonlinearity=nonlinearities.softmax,

    # learning rate parameters
    update_learning_rate=0.01,
    update_momentum=0.9,
    regression=False,
    # We only train for 10 epochs
    max_epochs=10,
    verbose=1,

    # Training test-set split
    eval_size = 0.2
    )

net = net1.fit(X[0:1000,:,:,:],y[0:1000])

net.predict(X[3000:3010,:,:,:])

import cPickle as pickle
with open('data/net1.pickle', 'wb') as f:
    pickle.dump(net, f, -1)

get_ipython().magic('ls -rtlh data')

import cPickle as pickle
with open('data/net1.pickle', 'rb') as f:
    net_pretrain = pickle.load(f)

net_pretrain.fit(X[0:1000,:,:,:],y[0:1000]);

net_pretrain.max_epochs = 5
net_pretrain.fit(X[1000:2000,:,:,:],y[1000:2000]);

toTest = range(3001,3026)
preds = net1.predict(X[toTest,:,:,:])
preds

fig = plt.figure(figsize=(10,10))
for i,num in enumerate(toTest):
    a=fig.add_subplot(5,5,(i+1)) #NB the one based API sucks!
    plt.axis('off')
    a.set_title(str(preds[i]) + " (" + str(y[num]) + ")")
    plt.imshow(-X[num,0,:,:], interpolation='none',cmap=plt.get_cmap('gray'))

import operator
import numpy as np
weights = [w.get_value() for w in net.get_all_params()]
numParas = 0
for i, weight in enumerate(weights):
    n = reduce(operator.mul, np.shape(weight))
    print(str(i), " ", str(np.shape(weight)), str(n))  
    numParas += n
print("Number of parameters " + str(numParas))

conv = net.get_all_params()

ws = conv[7].get_value() #Use the layernumber for the '(32, 1, 3, 3)', '288' layer from above
fig = plt.figure(figsize=(6,6))
for i in range(0,32):
    a=fig.add_subplot(6,6,(i+1))#NB the one based API sucks! 
    plt.axis('off')
    plt.imshow(ws[i,0,:,:], interpolation='none',cmap=plt.get_cmap('gray'))

