import matplotlib.image as mpimg

# First, load the image
filename = "images/MarshOrchid.jpg"
image = mpimg.imread(filename)
print image

# Print out its shape
print(image.shape)

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
plt.imshow(image)
plt.show()

import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# First, load the image again
filename = "images/MarshOrchid.jpg"
image = mpimg.imread(filename)

# Create a Tensorflow Variable
x = tf.Variable(image, name='x')

model = tf.initialize_all_variables()

with tf.Session() as session:
    x = tf.transpose(x, perm=[1, 0, 2])
    session.run(model)
    result = session.run(x)
        
plt.imshow(result)
plt.show()

get_ipython().magic('matplotlib inline')
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# First, load the image again
filename = "images/MarshOrchid.jpg"
image = mpimg.imread(filename)
height, width, depth = image.shape

# Create a TensorFlow Variable
x = tf.Variable(image, name='x')

model = tf.initialize_all_variables()

with tf.Session() as session:
    x = tf.reverse_sequence(x, [width] * height, 1, batch_dim=0)
    session.run(model)
    result = session.run(x)

print(result.shape)
plt.imshow(result)
plt.show()

x = tf.reverse_sequence(x, np.ones((height,)) * width, 1, batch_dim=0)

