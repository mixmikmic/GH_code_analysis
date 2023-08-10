get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as imgplot
from load_mnist import load_data_2d

X,y,PIXELS = load_data_2d('../../data/mnist.pkl.gz')

X.shape,y.shape

np.mean(X[0,0,:,:]),np.mean(X[:,0,1,1]),np.mean(X[:,0,2,1]),np.var(X[:,0,10,10])

y[0:10]

fig = plt.figure(figsize=(10,30))
for i in range(3):
    a=fig.add_subplot(1,3,(i+1))
    plt.imshow(X[i,0,:,:], interpolation='none',cmap=plt.get_cmap('gray'))

from lasagne import layers
from lasagne import nonlinearities
from nolearn.lasagne import NeuralNet

net1 = NeuralNet(
    # Geometry of the network
    layers=[
        ('input', layers.InputLayer),
        ('hidden1', layers.DenseLayer),
        ('hidden2', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, PIXELS, PIXELS), #None in the first axis indicates that the batch size can be set later
    hidden1_num_units=500,
    hidden2_num_units=50,
    output_num_units=10, output_nonlinearity=nonlinearities.softmax,

    # learning rate parameters
    update_learning_rate=0.01,
    update_momentum=0.9,
    regression=False,
    # We only train for 10 epochs
    max_epochs=100,
    verbose=1,

    # Training test-set split
    eval_size = 0.2
    )

net = net1.fit(X[0:1000,:,:,:],y[0:1000])

toTest = range(1001,1011)
preds = net1.predict(X[toTest,:,:,:])
preds

fig = plt.figure(figsize=(10,100))
for i,num in enumerate(toTest):
    a=fig.add_subplot(1,10,(i+1)) #NB the one based API sucks!
    a.set_title(str(y[num]) + " / " + str(preds[i]))
    plt.imshow(X[num,0,:,:], interpolation='none',cmap=plt.get_cmap('gray'))



