import numpy as np

#image tools
from matplotlib import pyplot as plt
from PIL import Image

get_ipython().magic('matplotlib inline')



from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print ("The Training set is:")
print (mnist.train.images.shape)
print (mnist.train.labels.shape)
print ("The Test set is:")
print (mnist.test.images.shape)
print (mnist.test.labels.shape)
print ("Validation Set?")
print (mnist.validation.images.shape)
print (mnist.validation.labels.shape)

mnist.train.images[5]

# well, that's not a picture, it's an array.

mnist.train.images[5].shape

a = np.reshape(mnist.train.images[5], [28,28])

# So now we have a 28x28 matrix, where each element is an intensity level from 0 to 1.  
a.shape

plt.imshow(a, cmap='Greys_r')



