a = (1,2,3,4)
a[::-1]

#We load the net and the data from the last example
from lasagne import layers
from lasagne import nonlinearities
from nolearn.lasagne import NeuralNet
import cPickle as pickle
import gzip
with gzip.open('mnist_4000.pkl.gz', 'rb') as f:
    (X,y) = pickle.load(f)
PIXELS = len(X[0,0,0,:])
X.shape, y.shape, PIXELS

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
        conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2), #pool_size used to be called ds in old versions of lasagne
        conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
        hidden4_num_units=500,
        output_num_units=10, output_nonlinearity=nonlinearities.softmax,

        # learning rate parameters
        update_learning_rate=0.01,
        update_momentum=0.90,
        regression=False,
        # We only train for 10 epochs
        max_epochs=10,
        verbose=1,

        # Training test-set split
        eval_size = 0.2
        )

net1 = createNet()
d = net1.fit(X[0:1000,:,:,:],y[0:1000]); #Training with the 

from nolearn.lasagne import BatchIterator

class SimpleBatchIterator(BatchIterator):

    def transform(self, Xb, yb):
        # The 'incomming' and outcomming shape is (10, 1, 28, 28)
        Xb, yb = super(SimpleBatchIterator, self).transform(Xb, yb)
        return Xb[:,:,:,::-1], yb #<--- Here we do the flipping of the images

# Setting the new batch iterator
net1.batch_iterator_train = SimpleBatchIterator(batch_size=10)
d = net1.fit(X[0:1000,:,:,:],y[0:1000])

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib.image as imgplot
fig = plt.figure(figsize=(10,10))
for i in range(18):
    a=fig.add_subplot(6,6,2*i+1,xticks=[], yticks=[])
    plt.imshow(-X[i,0,:,:], cmap=plt.get_cmap('gray'))
    a=fig.add_subplot(6,6,2*i+2,xticks=[], yticks=[])
    plt.imshow(-X[i,0,:,::-1], cmap=plt.get_cmap('gray'))

