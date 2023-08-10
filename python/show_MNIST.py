import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

MNIST = input_data.read_data_sets("MNIST_data/", one_hot=True)

def image(imnumber):
    a = MNIST.train.images[imnumber, 0:784]
    b = np.fliplr(a.reshape((28, 28)))
    c = MNIST.train.labels[imnumber]
    x = np.linspace(imnumber, 28, 28)
    y = x
    print('Label:', int(np.nonzero(c)[0]))
    plt.pcolormesh(x, y, b, cmap='Greys')
    plt.show()

print('Enter -1 to exit')
while (True):
    print('')
    imnumber = int(input('Enter Image Number: '))
    if (imnumber == -1):
        break

    image(imnumber)



