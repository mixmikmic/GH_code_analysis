dataSize = 2000
epochs = 200

from load_mnist import load_data_2d
X,y,PIXELS = load_data_2d('../../data/mnist.pkl.gz')

import numpy as np

#maxs = np.max(X[:,0,:,:],axis=(1,2))
#mins = np.min(X[:,0,:,:],axis=(1,2))
#Xs = np.zeros_like(X)
#for i in range(len(X)):
#    Xs[i,0,:,:] = (X[i,0,:,:] - maxs[i])/(maxs[i]-mins[i])
#X = Xs

Xs = (X - np.min(X)) / (np.max(X) - np.min(X))
X = Xs * 256.0    
np.min(X),np.max(X)

from lasagne import layers
from lasagne import nonlinearities
from nolearn.lasagne import NeuralNet

def createNet():
   return NeuralNet(
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
        update_momentum=0.90,
        regression=False,
        # We only train for 10 epochs
        max_epochs=epochs,
        verbose=1,

        # Training test-set split
        eval_size = 0.2
        )

netnoAug = createNet()
d = netnoAug.fit(X[0:dataSize,:,:,:],y[0:dataSize]); #Training with only 1000 examples

get_ipython().magic('matplotlib inline')
import pandas as pd
dfNoAug = pd.DataFrame(netnoAug.train_history_)
dfNoAug[['train_loss','valid_loss','valid_accuracy']].plot(title='No Augmentation')

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib.image as imgplot
import numpy as np
from skimage import transform as tf

rots = np.asarray((-6,-5,-4,-3,3,4,5,6)) / (360 / (2.0 * np.pi))
dists = (-2,-1,1,2)

def manipulateTrainingData(Xb):
    retX = np.zeros((Xb.shape[0], Xb.shape[1], Xb.shape[2], Xb.shape[3]), dtype='float32')
    for i in range(len(Xb)):
        dist = dists[np.random.randint(0, len(dists))]
        rot = rots[np.random.randint(0, len(rots))]
        scale = np.random.uniform(0.9,1.10)
        tform = tf.SimilarityTransform(rotation=rot, translation=dist, scale=scale)
        retX[i,0,:,:] = 256.0 * tf.warp(Xb[i,0,:,:]/256.0,tform) # "Float Images" are only allowed to have values between -1 and 1
    return retX


Xb = np.copy(Xs[0:100,:,:,:])
Xb = manipulateTrainingData(Xb)

fig = plt.figure(figsize=(20,150))
for i in range(10):
    a=fig.add_subplot(1,20,2*i+1,xticks=[], yticks=[])
    plt.imshow(-Xs[i,0,:,:], cmap=plt.get_cmap('gray'))
    a=fig.add_subplot(1,20,2*i+2,xticks=[], yticks=[])
    plt.imshow(-Xb[i,0,:,:], cmap=plt.get_cmap('gray'))    

from nolearn.lasagne import BatchIterator

class SimpleBatchIterator(BatchIterator):

    def transform(self, Xb, yb):
        # The 'incomming' and outcomming shape is (10, 1, 28, 28)
        Xb, yb = super(SimpleBatchIterator, self).transform(Xb, yb)
        return manipulateTrainingData(Xb), yb #<--- Here we do the manipulations of the training set

# Setting the new batch iterator
net1Aug = createNet()
net1Aug.batch_iterator_train = SimpleBatchIterator(100)
d = net1Aug.fit(X[0:dataSize,:,:,:],y[0:dataSize])

dfAug = pd.DataFrame(net1Aug.train_history_)
dfAug[['train_loss','valid_loss','valid_accuracy']].plot(title='With Augmentation')

get_ipython().magic('load_ext rpy2.ipython')
get_ipython().magic('Rpush dfAug')
get_ipython().magic('Rpush dfNoAug')

get_ipython().run_cell_magic('R', '', "library(ggplot2)\nggplot() + aes(x=epoch, colour='Loss') + \n  geom_line(data=dfAug, aes(y = train_loss, colour='Training'), size=2) + \n  geom_line(data=dfAug, aes(y = valid_loss, colour='Validation'), size=2) + \n  geom_line(data=dfNoAug, aes(y = train_loss, colour='Training'), size=2) + \n  geom_line(data=dfNoAug, aes(y = valid_loss, colour='Validation'), size=2) + \n  xlab('Epochs') + ylab('Loss') +\n  ylim(c(0,0.75))")

get_ipython().run_cell_magic('R', '', "library(ggplot2)\nggplot() + aes(x=epoch, colour='Loss') + \n  geom_line(data=dfAug, aes(y = valid_accuracy, colour='Augmented'), size=2) + \n  geom_line(data=dfNoAug, aes(y = valid_accuracy, colour='Not Augmented'), size=2) + \n  xlab('Epochs') + ylab('Accuracy on validation set') +\n  ylim(c(0.75,1))")

