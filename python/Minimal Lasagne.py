get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib.image as imgplot
from load_mnist import load_data_2d

X,y,PIXELS = load_data_2d('../../data/mnist.pkl.gz')

X.shape

y.shape

y[0:10]

fig = plt.figure(figsize=(20,100))
for i in range(5):
    a=fig.add_subplot(1,5,i)
    plt.imshow(X[i,0,:,:], interpolation='none',cmap=plt.get_cmap('gray'))

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
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_ds=(2, 2),
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_ds=(2, 2),
    hidden4_num_units=500,
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

net = net1.fit(X[0:50000,:,:,:],y[0:50000])

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

preds = net_pretrain.predict(X[3000:3010,:,:,:])
preds, y[3000:3010],X[3000:3010,:,:,:].shape

fig = plt.figure()
for i in range(10):
    a=fig.add_subplot(1,10,(i+1)) #NB the one based API sucks!
    a.set_title(str(y[(3000 + i)]) + " / " + str(preds[i]))
    plt.imshow(X[(3000+i),0,:,:], cmap=plt.get_cmap('gray'))



