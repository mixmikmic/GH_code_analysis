import numpy as np
import chainer

# Load the MNIST dataset from pre-inn chainer method
train, test = chainer.datasets.get_mnist()

# train[i] represents i-th data, there are 60000 training data
# test data structure is same, but total 10000 test data
print('len(train), type ', len(train), type(train))
print('len(test), type ', len(test), type(test))

print('train[0]', type(train[0]), len(train[0]))
# print(train[0])  # x_i = long array and y_i = label

# train[i][0] represents x_i, MNIST image data,
# type=numpy(784,) vector <- specified by ndim of get_mnist()
print('train[0][0]', train[0][0].shape)
np.set_printoptions(threshold=10)  # set np.inf to print all.
print(train[0][0])

# train[i][1] represents y_i, MNIST label data(0-9), type=numpy() -> this means scalar
print('train[0][1]', train[0][1].shape, train[0][1])

import os

import chainer
import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

base_dir = 'src/02_mnist_mlp/images'

# Load the MNIST dataset from pre-inn chainer method
train, test = chainer.datasets.get_mnist(ndim=1)

ROW = 4
COLUMN = 5
for i in range(ROW * COLUMN):
    # train[i][0] is i-th image data with size 28x28
    image = train[i][0].reshape(28, 28)   # not necessary to reshape if ndim is set to 2
    plt.subplot(ROW, COLUMN, i+1)          # subplot with size (width 3, height 5)
    plt.imshow(image, cmap='gray')  # cmap='gray' is for black and white picture.
    # train[i][1] is i-th digit label
    plt.title('label = {}'.format(train[i][1]))
    plt.axis('off')  # do not show axis value
plt.tight_layout()   # automatic padding between subplots
plt.savefig(os.path.join(base_dir, 'mnist_plot.png'))
#plt.show()



