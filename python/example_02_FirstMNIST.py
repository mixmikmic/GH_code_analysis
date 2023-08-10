import matplotlib.pyplot as plt
import numpy as np
import sys, math
sys.path.insert(1,'../'); #print sys.path # To import mist_load
from mnist_loader import *
from Network import *

training_data, validation_data, test_data = load_data_wapper('../data/mnist.pkl.gz')
n_imgs_train = np.shape(training_data)[0]
n_imgs_val   = np.shape(validation_data)[0]
n_imgs_test  = np.shape(test_data)[0]
n_pixels     = len(training_data[0][0])
n_class      = len(training_data[0][1])

print 'Number of training images   %d'%( n_imgs_train )
print 'Number of validation images %d'%( n_imgs_val )
print 'Number of test images       %d'%( n_imgs_test )
print 'Number of pixels            %d'% n_pixels
print 'Number of class             %d'% n_class

n_shows = 20
fig = plt.figure(figsize=(25,5))
for i in range(0,n_shows):
    X = training_data[i][0]
    fig.add_subplot( math.ceil(n_shows/10), 10, i+1)
    plt.imshow(X.reshape(28,28), cmap ='binary')
plt.show()

n_hidenLayer = 30
net = Network([n_pixels, n_hidenLayer, n_class])

epochs = 30
mini_batch_size = 10
net.SGD(training_data, 
        epochs=epochs, 
        mini_batch_size=mini_batch_size, 
        eta=3.0,
        test_data=test_data)



